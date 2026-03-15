# TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery

This is the Pytorch Implementation of [**_TempME:Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery_**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html) [[arXiv]](https://arxiv.org/abs/2310.19324)


### Train a Base Model

To start, you'll need to train a base model. Our framework supports several base model types, including TGAT, TGN, and GraphMixer. To train your model, use the following command, replacing `${type}` with your chosen base model type (e.g., `tgat`, `tgn`, `graphmixer`) and `${dataset}` with the name of your dataset.

```bash
python learn_base.py --base_type ${type} --data ${dataset}
```


### Train an Explainer
Once you have a base model, the next step is to train an explainer. Use the following command to train your explainer:

```bash
python temp_exp_main.py --base_type ${type} --data ${dataset}
```

### Verify Enhancement Effect
To evaluate the effectiveness of the explanatory subgraphs extracted by the explainer, use the following command:

```bash
python enhance_main.py --data ${dataset} --base_type ${type}
```
## Citation
If you find this work useful, please consider citing:

```
@article{chen2024tempme,
  title={Tempme: Towards the explainability of temporal graph neural networks via motif discovery},
  author={Chen, Jialin and Ying, Rex},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```