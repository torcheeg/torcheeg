'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const folder = path.join(__dirname, '../../docs/source/_static/leaderboard');
const api = require(path.join(folder, 'leaderboard.js'));
const data = JSON.parse(fs.readFileSync(path.join(folder, 'results.json'), 'utf8'));

test('filters combine task, model, evidence, preprocessing and case-insensitive search', () => {
  assert.equal(api.filterResults(data, {task: 'Classification'}).length, 2);
  assert.equal(api.filterResults(data, {model: 'EEGNet', evidence: 'validation'}).length, 1);
  assert.equal(api.filterResults(data, {query: 'ctNET'}).length, 1);
  assert.equal(api.filterResults(data, {query: 'no matching experiment'}).length, 0);
  assert.equal(api.filterResults(data, {evidence: 'published'}).length, 0);
  const protocol = data.protocols[data.results[0].protocol];
  assert.equal(api.filterResults(data, {preprocessing: protocol.preprocessing}).length, 2);
  assert.equal(api.filterResults(data, {model: 'EEGNet', task: 'Feature-grid generation'}).length, 0);
});

test('ranking is confined to matching protocols, metric units and directions', () => {
  const groups = api.groupResults(data.results);
  assert.equal(groups.length, 2);
  const classification = groups.find(rows => rows[0].metric.name === 'accuracy');
  assert.equal(classification.length, 2);
  assert.ok(classification[0].metric.value >= classification[1].metric.value);
  assert.throws(() => api.rankResults(data.results), /different protocols or metrics/);
  const mixed = structuredClone(classification);
  mixed[1].metric.unit = 'other';
  assert.throws(() => api.rankResults(mixed), /different protocols or metrics/);
});

test('lower-is-better sorting preserves the original records', () => {
  const first = structuredClone(data.results.find(row => row.metric.direction === 'lower'));
  const second = structuredClone(first);
  second.id = 'another-run';
  second.metric.value = first.metric.value / 2;
  const rows = [first, second];
  assert.equal(api.rankResults(rows)[0], second);
  assert.equal(rows[0], first);
  assert.deepEqual(api.rankResults([]), []);
});

test('metrics and source links retain their scientific meaning', () => {
  assert.equal(api.formatMetric({value: 0.75, unit: 'fraction'}), '75.00%');
  assert.equal(api.formatMetric({value: 0.75, std: 0.01, unit: 'fraction'}), '75.00% ± 1.00%');
  assert.equal(api.formatMetric({value: 0.123456, unit: 'score'}), '0.12346');
  data.results.forEach(row => {
    assert.equal(api.exampleURL(row), 'https://github.com/torcheeg/torcheeg/blob/' +
      row.implementation.commit + '/' + row.implementation.example);
  });
});
