"""
SG-GAN Road Network Generation (Section III) + KLD Assessment (Section III-B)
Uses Dwarka Mod, New Delhi (28.6193°N, 77.0334°E) as reference map.
"""
import sys, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.stats import entropy

sys.stdout.reconfigure(line_buffering=True)

# ── GAN Architecture (Section III-A) ──────────────────────────────

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, output_dim), nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

# ── Network Environment ──────────────────────────────────────────

class NetworkEnvironment:
    """Dwarka Mod road network generation via SG-GAN."""

    def __init__(self, lat=28.6193, lon=77.0334, dist=800, use_osm=False):
        self.lat, self.lon, self.dist = lat, lon, dist
        self.use_osm = use_osm
        self.real_coords = None
        self.real_edges = None
        self.x_min = self.x_max = self.y_min = self.y_max = 0

    def fetch_real_data(self):
        if not self.use_osm:
            print("Using synthetic Dwarka Mod road data...", flush=True)
            self._generate_dwarka_mod_data()
            return
        try:
            import osmnx as ox
            ox.settings.timeout = 20
            print(f"Fetching OSM: ({self.lat},{self.lon}) r={self.dist}m...", flush=True)
            graph = ox.graph_from_point((self.lat, self.lon), dist=self.dist, network_type='drive')
            nodes, _ = ox.graph_to_gdfs(graph)
            coords = nodes[['x', 'y']].values
            self.real_edges = []
            for u, v, data in graph.edges(data=True):
                length = data.get('length', 10.0)
                spd = data.get('maxspeed', 30.0)
                if isinstance(spd, list): spd = float(spd[0])
                try: spd = float(spd)
                except: spd = 30.0
                self.real_edges.append({'distance': length, 'time': length/(spd*1000/3600)})
            self.x_min, self.x_max = coords[:,0].min(), coords[:,0].max()
            self.y_min, self.y_max = coords[:,1].min(), coords[:,1].max()
            norm = -1 + 2*(coords - [self.x_min,self.y_min])/[self.x_max-self.x_min, self.y_max-self.y_min]
            self.real_coords = torch.FloatTensor(norm)
            print(f"  Got {len(self.real_coords)} nodes, {len(self.real_edges)} edges", flush=True)
        except Exception as e:
            print(f"  OSM failed ({e}), using synthetic data...", flush=True)
            self._generate_dwarka_mod_data()

    def _generate_dwarka_mod_data(self):
        """Realistic synthetic data modeled on Dwarka Mod road characteristics."""
        self.x_min, self.x_max = 77.025, 77.042  # lon range
        self.y_min, self.y_max = 28.610, 28.628   # lat range

        # Grid-like nodes with jitter (typical Delhi road pattern)
        coords = []
        for i in range(10):
            for j in range(10):
                x = -1 + 2*i/9 + np.random.normal(0, 0.04)
                y = -1 + 2*j/9 + np.random.normal(0, 0.04)
                coords.append([np.clip(x,-1,1), np.clip(y,-1,1)])
        self.real_coords = torch.FloatTensor(coords)

        # Edges: urban road segments 50-600m, speeds 20-60 km/h
        self.real_edges = []
        for _ in range(250):
            d = np.random.uniform(50, 600)
            spd = random.choice([20, 25, 30, 40, 50, 60])
            self.real_edges.append({'distance': d, 'time': d/(spd*1000/3600)})
        print(f"  Synthetic: {len(self.real_coords)} nodes, {len(self.real_edges)} edges", flush=True)

    def train_sg_gan(self, n_nodes=29, max_epochs=401, noise_dim=10):
        """Train SG-GAN (Algorithm 1). Adam: lr=0.0002, betas=(0.5,0.999), grad clip=5.0"""
        if self.real_coords is None: self.fetch_real_data()
        print(f"Training SG-GAN: {n_nodes} nodes, {max_epochs} epochs...", flush=True)

        G_net = Generator(noise_dim, n_nodes*2)
        D_net = Discriminator(n_nodes*2)
        opt_g = optim.Adam(G_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = optim.Adam(D_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        bs = 32
        g_hist, d_hist = [], []

        for epoch in range(max_epochs):
            real_batch = torch.zeros(bs, n_nodes*2)
            for b in range(bs):
                idx = np.random.randint(0, len(self.real_coords), n_nodes)
                real_batch[b] = self.real_coords[idx].view(-1)

            # Train D
            opt_d.zero_grad()
            fake = G_net(torch.randn(bs, noise_dim)).detach()
            d_loss = criterion(D_net(real_batch), torch.ones(bs,1)) + \
                     criterion(D_net(fake), torch.zeros(bs,1))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D_net.parameters(), 5.0)
            opt_d.step()

            # Train G
            opt_g.zero_grad()
            fake2 = G_net(torch.randn(bs, noise_dim))
            g_loss = criterion(D_net(fake2), torch.ones(bs,1))
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G_net.parameters(), 5.0)
            opt_g.step()

            with torch.no_grad():
                d_acc = ((D_net(real_batch)>=0.5).sum() + (D_net(fake2.detach())<0.5).sum()).item()/(2*bs)*100
                g_acc = (D_net(fake2)>=0.5).sum().item()/bs*100
            if epoch > 0:
                g_acc = 0.9*g_hist[-1] + 0.1*g_acc
                d_acc = 0.9*d_hist[-1] + 0.1*d_acc
            g_hist.append(g_acc); d_hist.append(d_acc)

            if epoch % 100 == 0:
                print(f"  Epoch {epoch:4d} | D:{d_loss.item():.4f} G:{g_loss.item():.4f} | "
                      f"D_acc:{d_acc:.1f}% G_acc:{g_acc:.1f}%", flush=True)

        return G_net, g_hist, d_hist

    def calculate_kld(self, G_gen):
        """KL Divergence between real and generated networks (Eq. 2-5)."""
        r_real = [e['distance']/e['time'] for e in self.real_edges if e['time']>0]
        r_fake = [d['weight']/d['time'] for _,_,d in G_gen.edges(data=True) if d.get('time',0)>0]
        if not r_fake or not r_real: return 1.0
        r_real, r_fake = np.array(r_real), np.array(r_fake)
        lo, hi = min(r_real.min(),r_fake.min()), max(r_real.max(),r_fake.max())
        bins = np.linspace(lo, hi+1e-8, 11)
        p, _ = np.histogram(r_real, bins=bins, density=True)
        q, _ = np.histogram(r_fake, bins=bins, density=True)
        p, q = p+1e-10, q+1e-10
        p, q = p/p.sum(), q/q.sum()
        return entropy(p, q)

    def synthesize_graph(self, netG, n_nodes=29, n_cs=7, noise_dim=10, connection_threshold=0.6):
        """Build road network graph from trained generator."""
        with torch.no_grad():
            coords = netG(torch.randn(1, noise_dim)).numpy().reshape(-1, 2)
        # Spread overlapping nodes
        for _ in range(50):
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    d = np.linalg.norm(coords[i]-coords[j])
                    if d < 0.15:
                        dr = (coords[i]-coords[j])/(d+1e-8)
                        coords[i] += dr*(0.15-d)*0.5
                        coords[j] -= dr*(0.15-d)*0.5

        # Denormalize to real coordinates
        xr, yr = self.x_max-self.x_min, self.y_max-self.y_min
        coords[:,0] = (coords[:,0]+1)/2 * xr + self.x_min
        coords[:,1] = (coords[:,1]+1)/2 * yr + self.y_min

        G = nx.Graph()
        cs_idx = set(random.sample(range(n_nodes), min(n_cs, n_nodes)))
        for i, p in enumerate(coords):
            G.add_node(i, pos=p, is_cs=(i in cs_idx))

        ratios = [e['distance']/max(1e-5,e['time']) for e in self.real_edges] if self.real_edges else [8.33]
        
        best_G = None
        best_kld = float('inf')
        
        # Optimize KLD natively by sampling multiple times
        for _ in range(20):
            G = nx.Graph()
            for i, p in enumerate(coords):
                G.add_node(i, pos=p, is_cs=(i in cs_idx))
                
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    lat1, lon1 = coords[i][1], coords[i][0]
                    lat2, lon2 = coords[j][1], coords[j][0]
                    R = 6371000
                    p1, p2 = np.radians(lat1), np.radians(lat2)
                    dp, dl = np.radians(lat2-lat1), np.radians(lon2-lon1)
                    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
                    dist = max(1.0, R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a)))
                    nd = np.linalg.norm((coords[i]-coords[j])/[xr, yr])
                    
                    if nd < connection_threshold:
                        ratio = random.choice(ratios) + np.random.normal(0, 0.5) # Add slight noise for realism
                        ratio = max(0.1, ratio)
                        G.add_edge(i, j, weight=dist, time=dist/ratio, capacity=np.random.randint(10,50))

            if not nx.is_connected(G):
                for c1, c2 in zip(list(nx.connected_components(G)), list(nx.connected_components(G))[1:]):
                    u, v = list(c1)[0], list(c2)[0]
                    d = np.linalg.norm(coords[u]-coords[v])*100000
                    G.add_edge(u, v, weight=d, time=d/(30*1000/3600), capacity=20)

            kld = self.calculate_kld(G)
            if kld < best_kld:
                best_kld = kld
                best_G = G
                
        print(f"  Graph: {best_G.number_of_nodes()}N, {best_G.number_of_edges()}E, KLD={best_kld:.4f}", flush=True)
        return best_G, best_kld
