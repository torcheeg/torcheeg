#!/usr/bin/env python3
"""Generate model-selection SVGs using only the Python standard library.

Run from any directory: python docs/tools/generate_split_diagrams.py
All illustrations share a 1200 x 480 canvas and fixed-size sample marks.
The diagrams illustrate membership, not temporal order or default split ratios.
"""
from html import escape
from pathlib import Path

WIDTH, HEIGHT = 1200, 480
TRAIN, TEST, INK, MUTED = '#B6BDC8', '#A77BCB', '#253047', '#637086'
# stem, membership unit, per-subject experiment, folds, number of subjects
SPECS = [
    ('KFold', 'sample', False, 3, 3),
    ('KFoldCrossSubject', 'subject', False, 2, 4),
    ('KFoldCrossTrial', 'trial', False, 3, 3),
    ('KFoldGroupbyTrial', 'within', False, 3, 3),
    ('KFoldPerSubject', 'sample', True, 3, 1),
    ('KFoldPerSubjectCrossTrial', 'trial', True, 3, 1),
    ('KFoldPerSubjectGroupbyTrial', 'within', True, 3, 1),
    ('LeaveOneSubjectOut', 'subject', False, 3, 3),
    ('train_test_split', 'sample', False, 1, 3),
    ('train_test_split_cross_subject', 'subject', False, 1, 3),
    ('train_test_split_cross_trial', 'trial', False, 1, 3),
    ('train_test_split_groupby_trial', 'within', False, 1, 3),
    ('train_test_split_per_subject_cross_trial', 'trial', True, 1, 1),
    ('train_test_split_per_subject_groupby_trial', 'within', True, 1, 1),
]


def text(parts, x, y, value, size=16, fill=INK, anchor='start', weight='400'):
    parts.append('<text x="%s" y="%s" font-size="%s" fill="%s" '
                 'text-anchor="%s" font-weight="%s">%s</text>' %
                 (x, y, size, fill, anchor, weight, escape(value)))


def block(parts, x, y, held_out):
    parts.append('<rect x="%s" y="%s" width="16" height="18" rx="3" fill="%s"/>' %
                 (x, y, TEST if held_out else TRAIN))
    if held_out:  # A second visual cue, independent of hue.
        parts.append('<path d="M%s %s h8" stroke="white" stroke-width="2"/>' %
                     (x + 4, y + 9))


def held_out(stem, mode, fold, subject, trial, sample, folds):
    if mode == 'subject':
        if folds == 2:
            return subject // 2 == fold
        return subject == (fold if folds > 1 else 2)
    if mode == 'trial':
        # KFoldCrossTrial groups global trial IDs. The one-shot cross-trial
        # function partitions each subject's trials independently.
        selected = fold if folds > 1 else (subject + 2) % 3
        return trial == selected
    if mode == 'within':
        return sample // 2 == (fold if folds > 1 else 2)
    return sample // 4 == (fold if folds > 1 else 2)


def draw(spec):
    stem, mode, per_subject, folds, subjects = spec
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="480" '
             'viewBox="0 0 1200 480" role="img" aria-labelledby="title">',
             '<title id="title">%s</title>' % escape(stem),
             '<rect width="1200" height="480" fill="white"/>',
             '<g font-family="DejaVu Sans, Arial, sans-serif">']
    panel_w, gap = 352, 28
    start = (WIDTH - (folds * panel_w + (folds - 1) * gap)) / 2
    for fold in range(folds):
        x = start + fold * (panel_w + gap)
        if folds > 1:
            text(parts, x + 176, 115, 'Fold %s' % (fold + 1), 18, anchor='middle')
        if mode == 'sample':
            text(parts, x + 176, 217, 'Subject 1' if per_subject else 'Samples', 16, MUTED, 'middle')
            for sample in range(12):
                block(parts, x + 58 + sample * 20, 240,
                      held_out(stem, mode, fold, 0, 0, sample, folds))
        else:
            # Fixed-size marks on the common canvas, including one-subject plots.
            top = 158 + (4 - subjects) * 24
            for trial in range(3):
                text(parts, x + 126 + trial * 68, top - 15,
                     'Trial %s' % (trial + 1), 14, MUTED, 'middle')
            for subject in range(subjects):
                y = top + subject * 48
                text(parts, x + 88, y + 23, 'Subject %s' % (subject + 1), 14, MUTED, 'end')
                for trial in range(3):
                    for sample in range(6):
                        block(parts, x + 98 + trial * 68 + (sample % 3) * 20,
                              y + (sample // 3) * 21,
                              held_out(stem, mode, fold, subject, trial, sample, folds))
    block(parts, 499, 400, False)
    text(parts, 524, 415, 'Train', 16)
    block(parts, 625, 400, True)
    text(parts, 650, 415, 'Test', 16)
    parts.append('</g></svg>\n')
    return '\n'.join(parts)


def main():
    output = Path(__file__).resolve().parents[1] / 'source' / '_static' / 'splits'
    output.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        (output / (spec[0] + '.svg')).write_text(draw(spec), encoding='utf-8')
    print('Generated %s diagrams in %s' % (len(SPECS), output))


if __name__ == '__main__':
    main()
