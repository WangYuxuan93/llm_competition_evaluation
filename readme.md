## Example Format

The ground truth json file should contain the following entries:
```
[
    {
		"question_id": 1,
        "question": "What was the species richness value post-Hurricane Hugo in 1990?",
        "answer": "2.42 ± 0.14",
    },
    {
		"question_id": 2,
        "question": "For the vine _Philodendron scandens, what is the change in cover from Post-Hugo (1990) to Final (2010) mean cover?",
        "answer": "Increased from 1.07 ± 0.30 to 2.77 ± 0.70",
    }
]
```

## Prediction Format

The prediction json file should follow the format shown below:
```
{
"1": "This fluid is composed of water, sand or other proppants, and various chemical additives. The additives, which make up about 0.5% of the fracturing fluid, serve different purposes such as reducing friction, preventing scale formation, and inhibiting bacterial growth. The chemicals used can include substances like methanol, naphthalene, xylene, acetic acid, ammonia, and #2 fuel oil.",
"2": "less than 32.12MPa"
}
```

## Runnable Example

```
python evaluate.py --data_file text_retrieval_gold.json --pred_file text_retrieval_pred.json
```

```
python evaluate.py --data_file table_retrieval_gold.json --pred_file table_retrieval_pred.json
```

## Output Format
The output is a dict, where `exact` represents the Exact Match score, `f1` represents the F1 score, `overall` represents the overall score (average of EM and F1).
```
OrderedDict({'exact': 50.0, 'f1': 71.23893805309734, 'total': 2, 'overall': 60.61946902654867})
```