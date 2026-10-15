export type MapNode = {id: string; label: string; connections: number; indexed: boolean};
export type MapEdge = {id: string; source: string; target: string; similarity: number};
export type Particle = MapNode & {x: number; y: number; vx: number; vy: number; pinned: boolean};

// Repulsion separates notes, springs connect them, a weak center force keeps
// disconnected components visible. No tree hierarchy is imposed on the graph.
export function createParticles(nodes: MapNode[]): Particle[] {
  return nodes.map((node, index) => {
    const angle = index * Math.PI * (3 - Math.sqrt(5));
    const radius = 35 * Math.sqrt(index);
    return {...node, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius,
      vx: 0, vy: 0, pinned: false};
  });
}

export function stepLayout(nodes: Particle[], edges: MapEdge[], alpha: number) {
  const byId = new Map(nodes.map(node => [node.id, node]));
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    a.vx -= a.x * .0015 * alpha; a.vy -= a.y * .0015 * alpha;
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j], dx = a.x - b.x || .01, dy = a.y - b.y || .01;
      const distance = Math.max(10, Math.hypot(dx, dy));
      const force = 850 * alpha / (distance * distance);
      a.vx += dx / distance * force; a.vy += dy / distance * force;
      b.vx -= dx / distance * force; b.vy -= dy / distance * force;
    }
  }
  for (const edge of edges) {
    const a = byId.get(edge.source), b = byId.get(edge.target);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const distance = Math.max(.01, Math.hypot(dx, dy));
    const force = (distance - 105) * .022 * alpha;
    a.vx += dx / distance * force; a.vy += dy / distance * force;
    b.vx -= dx / distance * force; b.vy -= dy / distance * force;
  }
  for (const node of nodes) {
    if (node.pinned) {node.vx = 0; node.vy = 0; continue;}
    node.vx *= .72; node.vy *= .72;
    node.x += Math.max(-8, Math.min(8, node.vx));
    node.y += Math.max(-8, Math.min(8, node.vy));
  }
}
