'use strict';
// Stable square grids, packed in shelves. Coordinates retain the original node order.
function groupedSquareLayout(labels, aspect = 1) {
  const members = new Map();
  labels.forEach((label, index) => {
    if (!members.has(label)) members.set(label, []);
    members.get(label).push(index);
  });
  const groups = [...members].map(([label, indices]) => ({label, indices, side: Math.ceil(Math.sqrt(indices.length))}));
  groups.sort((a, b) => b.side - a.side || String(a.label).localeCompare(String(b.label), 'en', {numeric: true}));
  if (!groups.length) return {coords: [], groups: []};
  const gap = 2, area = groups.reduce((sum, group) => sum + (group.side + gap) ** 2, 0);
  const target = Math.sqrt(area * Math.max(0.25, aspect));
  let best;
  for (let attempt = 0; attempt < 20; attempt++) {
    const limit = Math.max(groups[0].side, target * (0.7 + attempt * 0.04));
    let x = 0, y = 0, rowHeight = 0, width = 0;
    const boxes = groups.map(group => {
      if (x && x + group.side > limit) { y += rowHeight + gap; x = 0; rowHeight = 0; }
      const box = {...group, x, y};
      width = Math.max(width, x + group.side); rowHeight = Math.max(rowHeight, group.side); x += group.side + gap;
      return box;
    });
    const height = y + rowHeight, score = Math.max(width / Math.max(0.25, aspect), height);
    if (!best || score < best.score) best = {groups: boxes, score};
  }
  const coords = Array(labels.length);
  for (const group of best.groups) group.indices.forEach((index, i) => {
    coords[index] = [group.x + i % group.side, -(group.y + Math.floor(i / group.side))];
  });
  return {coords, groups: best.groups};
}
