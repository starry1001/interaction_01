---
license: apache-2.0
task_categories:
- video-text-to-text
- visual-question-answering
language:
- en
tags:
- MLLM
- Evaluation
- Benchmark
- IWR
- Video2code
- multimodal
- code
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data.jsonl
dataset_info:
  features:
  - name: id
    dtype: int64
  - name: classification
    dtype:
      struct:
      - name: interaction_complexity
        dtype: string
      - name: visual_complexity
        dtype: string
      - name: domain
        dtype: string
      - name: sub_domain
        dtype: string
  - name: recording_details
    dtype:
      struct:
      - name: resolution
        dtype: string
      - name: environment
        dtype: string
  - name: source_website
    dtype: string
  - name: action_sequence
    sequence:
    - name: type
      dtype: string
    - name: parameters
      dtype:
        struct:
        - name: key
          dtype: string
        - name: description
          dtype: string
        - name: direction
          dtype: string
        - name: scroll_page
          dtype: string
    - name: visual_evaluation_flag
      dtype: bool
    - name: checkpoint_screenshot_path
      dtype: string
    - name: logical_assertion
      dtype: string
  - name: video_path
    dtype: string
  splits:
  - name: train
pretty_name: IWR-Bench
---


```code
@misc{chen2025iwrbenchlvlmsreconstructinteractive,
      title={IWR-Bench: Can LVLMs reconstruct interactive webpage from a user interaction video?}, 
      author={Yang Chen and Minghao Liu and Yufan Shen and Yunwen Li and Tianyuan Huang and Xinyu Fang and Tianyu Zheng and Wenxuan Huang and Cheng Yang and Daocheng Fu and Jianbiao Mei and Rong Wu and Yunfei Zhao and Licheng Wen and Xuemeng Yang and Song Mao and Qunshu Lin and Zhi Yu and Yongliang Shen and Yu Qiao and Botian Shi},
      year={2025},
      eprint={2509.24709},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.24709}, 
}
```