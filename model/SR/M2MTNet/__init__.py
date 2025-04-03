'''
@ARTICLE{M2MT-Net,
  author={Hu, Zeke Zexi and Chen, Xiaoming and Chung, Vera Yuk Ying and Shen, Yiran},
  journal={IEEE Transactions on Multimedia},
  title={Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-Resolution},
  year={2025},
  volume={27},
  number={},
  pages={1334-1348},
  doi={10.1109/TMM.2024.3521795}
}
'''
from .M2MTNet import M2MTNet as Network
from .M2MTNet import M2MTNetConfig as NetworkConfig


# For BasicLFSR
def get_model(args):
    return Network(scale=args.scale_factor, sz_a=[args.angRes_in, args.angRes_in], config=NetworkConfig())
