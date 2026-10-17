(function () {
  'use strict';

  const evidenceLabels = {published: 'Published', measured: 'Measured', validation: 'Validation run'};
  const metricLabels = {accuracy: 'Accuracy', reconstruction_mse: 'Reconstruction MSE'};

  function filterResults(data, filters) {
    const query = (filters.query || '').trim().toLowerCase();
    return data.results.filter(function (result) {
      const protocol = data.protocols[result.protocol];
      const fields = {dataset: protocol.dataset, task: protocol.task, model: result.model,
        preprocessing: protocol.preprocessing, evidence: result.evidence.kind};
      return Object.keys(fields).every(key => !filters[key] || filters[key] === fields[key]) &&
        (!query || [result.model, protocol.dataset, protocol.name, protocol.scope].join(' ').toLowerCase().includes(query));
    });
  }

  function comparisonKey(result) {
    return JSON.stringify([result.protocol, result.metric.name, result.metric.unit, result.metric.direction]);
  }

  function rankResults(results) {
    if (results.length === 0) return [];
    const key = comparisonKey(results[0]);
    if (results.some(result => comparisonKey(result) !== key)) {
      throw new Error('Results from different protocols or metrics cannot be ranked together.');
    }
    const sign = results[0].metric.direction === 'higher' ? -1 : 1;
    return results.slice().sort((a, b) => sign * (a.metric.value - b.metric.value) || a.model.localeCompare(b.model));
  }

  function groupResults(results) {
    const groups = new Map();
    results.forEach(function (result) {
      const key = comparisonKey(result);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(result);
    });
    return Array.from(groups.values()).map(rankResults);
  }

  function formatMetric(metric) {
    const format = value => metric.unit === 'fraction' ? (value * 100).toFixed(2) + '%' : value.toPrecision(5);
    return format(metric.value) + (metric.std === undefined || metric.std === null ? '' : ' ± ' + format(metric.std));
  }

  function exampleURL(result) {
    const source = result.implementation;
    return 'https://github.com/' + source.repository + '/blob/' + source.commit + '/' + source.example;
  }

  const api = {filterResults, rankResults, groupResults, formatMetric, exampleURL};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function renderGroup(data, results) {
    const first = results[0];
    const protocol = data.protocols[first.protocol];
    const section = element('section', undefined, 'leaderboard-group');
    section.append(element('h2', protocol.dataset + ': ' + protocol.name));
    section.append(element('p', protocol.scope, 'leaderboard-scope'));
    section.append(element('p', protocol.preprocessing));
    if (protocol.majority_accuracy !== undefined) {
      section.append(element('p', 'Majority-class accuracy on this test subset: ' +
        (protocol.majority_accuracy * 100).toFixed(2) + '% (always predict ' +
        (protocol.majority_label || 'the majority class') + ').'));
    }
    const wrap = element('div', undefined, 'leaderboard-table');
    const table = element('table', undefined, 'docutils align-default');
    table.append(element('caption', (metricLabels[first.metric.name] || first.metric.name) +
      ' — ' + first.metric.unit + '; ' + first.metric.direction + ' is better'));
    const head = element('thead');
    const headings = element('tr');
    ['Model', 'Value', 'Test windows', 'Evidence', 'Example and protocol'].forEach(function (text) {
      const cell = element('th', text);
      cell.scope = 'col';
      headings.append(cell);
    });
    head.append(headings);
    table.append(head);
    const body = element('tbody');
    results.forEach(function (result) {
      const row = element('tr');
      row.append(element('td', result.model));
      row.append(element('td', formatMetric(result.metric)));
      row.append(element('td', protocol.test_windows === null || protocol.test_windows === undefined ?
        'Not reported' : String(protocol.test_windows)));
      row.append(element('td', evidenceLabels[result.evidence.kind]));
      const references = element('td');
      const example = element('a', 'Example source');
      example.href = exampleURL(result);
      example.target = '_blank';
      example.rel = 'noopener noreferrer';
      references.append(example);
      if (result.evidence.citation_url) {
        const citation = element('a', 'Published source');
        citation.href = result.evidence.citation_url;
        citation.target = '_blank';
        citation.rel = 'noopener noreferrer';
        references.append(element('br'));
        references.append(citation);
      }
      const details = element('details');
      details.append(element('summary', 'Protocol, environment and provenance'));
      details.append(element('p', result.notes || ''));
      details.append(element('pre', JSON.stringify({protocol, metric: result.metric,
        evidence: result.evidence, implementation: result.implementation,
        environment: result.environment, configuration: result.configuration}, null, 2)));
      references.append(details);
      row.append(references);
      body.append(row);
    });
    table.append(body);
    wrap.append(table);
    section.append(wrap);
    return section;
  }

  async function initialize() {
    const root = document.getElementById('torcheeg-leaderboard');
    if (!root) return;
    const status = root.querySelector('[data-status]');
    try {
      const response = await fetch(root.dataset.source);
      if (!response.ok) throw new Error('Result download failed.');
      const data = await response.json();
      if (data.schema_version !== 1 || !Array.isArray(data.results)) throw new Error('Unsupported results.');
      const controls = root.querySelector('[data-controls]');
      const output = root.querySelector('[data-results]');
      const selects = {};
      const filterNames = {dataset: 'Dataset', task: 'Task', model: 'Model',
        preprocessing: 'Preprocessing', evidence: 'Evidence'};
      Object.keys(filterNames).forEach(function (key) {
        const label = element('label', filterNames[key]);
        const select = element('select');
        select.dataset.filter = key;
        select.setAttribute('aria-label', filterNames[key]);
        const all = element('option', 'All');
        all.value = '';
        select.append(all);
        const values = new Set(data.results.map(function (result) {
          const protocol = data.protocols[result.protocol];
          return key === 'model' ? result.model : key === 'evidence' ? result.evidence.kind : protocol[key];
        }));
        Array.from(values).sort().forEach(function (value) {
          const option = element('option', key === 'evidence' ? evidenceLabels[value] : value);
          option.value = value;
          select.append(option);
        });
        label.append(select);
        controls.append(label);
        selects[key] = select;
      });
      const searchLabel = element('label', 'Search');
      const search = element('input');
      search.type = 'search';
      search.setAttribute('aria-label', 'Search models and protocols');
      searchLabel.append(search);
      controls.append(searchLabel);
      const reset = element('button', 'Reset filters');
      reset.type = 'button';
      controls.append(reset);
      function update() {
        const filters = {query: search.value};
        Object.keys(selects).forEach(key => { filters[key] = selects[key].value; });
        const results = filterResults(data, filters);
        const groups = groupResults(results);
        output.replaceChildren(...groups.map(group => renderGroup(data, group)));
        status.textContent = results.length ? results.length + ' results in ' + groups.length + ' protocol/metric groups.' :
          'No results match these filters. Reset the filters to see all records.';
      }
      Object.values(selects).forEach(select => select.addEventListener('change', update));
      search.addEventListener('input', update);
      reset.addEventListener('click', function () {
        Object.values(selects).forEach(select => { select.value = ''; });
        search.value = '';
        update();
      });
      update();
    } catch (error) {
      status.textContent = 'Results are unavailable. Reload this page or use the JSON download below.';
    }
  }

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
    else initialize();
  }
}());
