## Generating data yourself 
Here you can see the helper functions I used to generate [SSOCR-V.1](https://huggingface.co/datasets/Ransaka/SSOCR-V.1). Once generated you can push these datasets into huggingface by using `push_to_hub.py` scripts. 

* If you have large dataset please use script [push_to_hub_v1.py](./push_to_hub_v1.py). 
* If you need faster image generation, try to use machine with high cpu count. With my setup (with 16 core), I was able to generate ~6000images/s.