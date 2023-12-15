# An Incremental Unified Framework for Small Defect Inspection

By Jiaqi Tang, Hao Lu, Xiaogang Xu, Ruizheng Wu, Sixing Hu, Tong Zhang, Tsz Wa Cheng, Ming Ge, Ying-Cong Chen* and Fugee Tsung

## Introduction
Artificial Intelligence (AI)-driven defect inspection is pivotal in industrial manufacturing. Yet, many methods, tailored to specific pipelines, grapple with diverse product portfolios and evolving processes. Addressing this, we present the Incremental Unified Framework (IUF) that can reduce the feature conflict problem when continuously integrating new objects in the pipeline, making it advantageous in object-incremental learning scenarios. Employing a state-of-the-art transformer, we introduce Object-Aware Self-Attention (OASA) to delineate distinct semantic boundaries. Semantic Compression Loss (SCL) is integrated to optimize non-primary semantic space, enhancing network adaptability for novel objects. Additionally, we prioritize retaining the features of established objects during weight updates. Demonstrating prowess in both image and pixel-level defect inspection, our approach achieves state-of-the-art performance, proving indispensable for dynamic and scalable industrial inspections.

![image](https://github.com/jqtangust/IUF/blob/ae8b9a5051a7308816fce8e3196acdc429d7d9d2/Motivation2.svg)

## Codes
*To Notice: Relevant code and materials will be released after the official publication.*

## Citation
```
@misc{tang2023incremental,
      title={An Incremental Unified Framework for Small Defect Inspection}, 
      author={Jiaqi Tang and Hao Lu and Xiaogang Xu and Ruizheng Wu and Sixing Hu and Tong Zhang and Tsz Wa Cheng and Ming Ge and Ying-Cong Chen and Fugee Tsung},
      year={2023},
      eprint={2312.08917},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


