<p align="center">
    <a href="https://github.com/jgamper/intrinsic-dimensionality/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/jgamper/intrinsic-dimensionality.svg?color=blue">
    </a>
    <a href="https://github.com/jgamper/intrinsic-dimensionality/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/jgamper/intrinsic-dimensionality?include_prereleases">
    </a>
</p>

* All contributions are welcome! Please raise an issue for a bug, feature or pull request!

* <a href="https://twitter.com/share?" class="twitter-share-button" data-text="Check this out!" data-url="https://github.com/jgamper/intrinsic-dimensionality" data-show-count="false">Tweet</a> about this repo!

* Give this repo a star! :star:

<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/intrinsic-dimensionality/master/docs/source/imgs/star_syntax.png?token=ADDZO4PH6CJSK5XTSC2ZLXK6ZPXRY" width="600"/>
<p>

# Quick Start

### Tissue mask and tiling pipeline
```python
from syntax.slide import Slide
from syntax.transformers.tissue_mask import OtsuTissueMask
from syntax.transformers.tiling import SimpleTiling
from syntax.transformers import Pipeline, visualize_pipeline_results

slide = Slide(slide_path=slide_path)
pipeline = Pipeline([OtsuTissueMask(), SimpleTiling(magnification=40,
                                                  tile_size=224,
                                                  max_per_class=10)])
slide = pipeline.fit_transform(slide)
visualize_pipeline_results(slide=slide,
                           transformer_list=pipeline.transformers,
                           title_list=['Tissue Mask', 'Random Tile Sampling'])
```
<p align="center">
    <img src="https://raw.githubusercontent.com/jgamper/compay-syntax/master/docs/source/imgs/simple_pipeline.png?token=ADDZO4ISOOTTRG4MMPNYCXS6ZPXPS" width="600"/>
<p>

# Install

`pip install compay-syntax==0.4.0`