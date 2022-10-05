Repo cloned from https://github.com/stanfordmlgroup/CheXbert

Minor modifications to code:

1) CheXbert/src/datasets renamed to CheXbert/src/datasets_chexbert to avoid naming conflicts
2) In function label of module **CheXbert/src/label.py**, I added **strict=False** in model.load_state_dict() (lines 70 and 77), otherwise there would be a state_dict missing key error when loading the bert model.<br>This is due to a version mismatch, since I use **transformers==4.19.2** and they used **transformers==2.5.1**
3) Also bert model weight is stored under CheXbert/src/models/chexbert.pth