// Mounts the project journey on a case-study page.
// It replaces the static pipeline figure in place; if anything here fails —
// no modules, no network, no WebGL — that figure simply stays as it was.
import { pipelines } from './pipelines.js';
import { buildJourney } from './pipeline.js';

const pipe = pipelines[document.body.dataset.pipeline];
if (pipe) {
  const journey = buildJourney(pipe);
  const viz = document.querySelector('.pipeline-viz');
  const block = viz && viz.closest('.fig-block');
  if (block) {
    block.parentNode.insertBefore(journey, block);
    block.hidden = true;               // superseded by the interactive version
  } else {
    // No pipeline figure on this page: open the methodology with the journey
    // rather than dropping it into the middle of whatever section came first.
    const heading = [...document.querySelectorAll('h2')]
      .find(h => /methodolog/i.test(h.textContent));
    const anchor = heading || document.querySelector('.fig-block') ||
                   document.querySelector('main section');
    if (anchor) anchor.parentNode.insertBefore(journey, anchor.nextSibling);
  }
}
