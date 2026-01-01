from mininet.topo import Topo

class FlowerTopo(Topo):
    """Custom topology for Hierarchical Flower FL.

    One core switch connects the cloud host and *N* per-server LAN switches.
    For each server *sid* we create:
      • leaf{sid}  – leaf server process
      • proxy{sid} – proxy client uploading to cloud
      • cli{sid}_{cid} – training clients (heterogeneous count)
    """

    def build(self, num_servers: int, clients_per_server: list[int]):
        assert len(clients_per_server) == num_servers, "clients_per_server length mismatch"

        # Single core switch
        core = self.addSwitch("s0")

        # Cloud aggregator host
        cloud = self.addHost("cloud")
        self.addLink(cloud, core)

        # Per-server LANs
        for sid in range(num_servers):
            lan = self.addSwitch(f"s{sid+1}")
            self.addLink(lan, core)

            # Leaf server & proxy client on this LAN
            leaf = self.addHost(f"leaf{sid}")
            proxy = self.addHost(f"proxy{sid}")
            self.addLink(leaf, lan)
            self.addLink(proxy, lan)

            # Training clients
            for cid in range(clients_per_server[sid]):
                cli = self.addHost(f"cli{sid}_{cid}")
                self.addLink(cli, lan)

# Required by the mnexec helper to locate the Topo from CLI
TOPO_CLASS = "FlowerTopo"
