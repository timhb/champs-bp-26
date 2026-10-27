'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {runInNewContext} = require('node:vm');
const context = {};
runInNewContext(readFileSync('notebooks/attention-viewer/assets/grouped-layout.js', 'utf8'), context);
const layout = context.groupedSquareLayout;

test('30,000 slides retain identity in disjoint compact square groups', () => {
  const labels = Array.from({length: 30000}, (_, i) => i < 15000 ? 'large' : `cluster ${i % 37}`);
  for (const aspect of [0.6, 1, 2.5]) {
    const result = layout(labels, aspect);
    assert.equal(result.coords.length, labels.length);
    assert.equal(new Set(result.coords.map(point => point.join(','))).size, labels.length);
    assert.equal(result.groups.length, 38);
    for (const group of result.groups) {
      assert.ok(group.indices.length <= group.side ** 2);
      assert.ok(group.indices.length > (group.side - 1) ** 2);
      for (const index of group.indices) {
        assert.equal(labels[index], group.label);
        const [x, y] = result.coords[index];
        assert.ok(Number.isFinite(x) && Number.isFinite(y));
        assert.ok(x >= group.x && x < group.x + group.side);
        assert.ok(-y >= group.y && -y < group.y + group.side);
      }
      for (const other of result.groups) {
        if (other === group) continue;
        assert.ok(group.x + group.side + 2 <= other.x || other.x + other.side + 2 <= group.x || group.y + group.side + 2 <= other.y || other.y + other.side + 2 <= group.y);
      }
    }
    const xs = result.coords.map(p => p[0]), ys = result.coords.map(p => p[1]);
    const footprint = (Math.max(...xs) - Math.min(...xs) + 1) * (Math.max(...ys) - Math.min(...ys) + 1);
    assert.ok(footprint < labels.length * 2, 'at least half of the packed area holds points');
    assert.equal(JSON.stringify(result), JSON.stringify(layout(labels, aspect)));
  }
});

test('empty, singleton, unassigned and many small groups', () => {
  for (const labels of [[], ['Unknown'], [-1, -1, 0, 1], Array.from({length: 1000}, (_, i) => i)]) {
    const result = layout(labels);
    assert.equal(result.coords.length, labels.length);
    assert.equal(new Set(result.coords.map(p => p.join(','))).size, labels.length);
    assert.equal(result.groups.length, new Set(labels).size);
  }
});
