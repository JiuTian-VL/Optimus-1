<div align="center">
<h2 align="center">
   <img src="./assets/optimus.png" style="vertical-align: middle; height: 1em; padding: 0 0.2em;"> <b>Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks</b> 
</h2>
<h3 align="center">
<b>NeurIPS 2024</b>
</h3>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=TDBF2UoAAAAJ&hl=en&oi=ao">Zaijing&#160;Li</a><sup>1 2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=KO77A2oAAAAJ&hl=en">Yuquan&#160;Xie</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=9Vc--XsAAAAJ&hl=en&oi=ao">Rui&#160;Shao</a><sup>1&#9993</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ&hl=en&oi=ao">Gongwei&#160;Chen</a><sup>1</sup>,
<br>
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=Awsue7sAAAAJ">Dongmei&#160;Jiang</a><sup>2</sup>,
 <a target="_blank" href="https://scholar.google.com/citations?hl=en&user=yywVMhUAAAAJ">Liqiang&#160;Nie</a><sup>1&#9993</sup>
</div>
<sup>1</sup>Harbin Institute of Technology,Shenzhen&#160&#160&#160</span>
<sup>2</sup>Peng Cheng Laboratory, Shenzhen</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>

<br/>
<br/>
<div align="center">
    <a href="https://arxiv.org/abs/2408.03615" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://cybertronagent.github.io/Optimus-1.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Optimus--1-9cf" alt="Project Page"></a>
</div>
</div>

<h3 align="center">
<b>:fire: code will release soon :smile:</b>
</h3>


## :new: Updates
- [09/2024] Optimus-1 is accepted to **NeurIPS2024**!
- [08/2024] [Arxiv paper](https://arxiv.org/abs/2408.03615) released.


## :balloon: Optimus-1 Framework
We divide the structure of Optimus-1 into Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. In a given game environment with a long-horizon task, the Knowledge-Guided Planner senses the environment, retrieves knowledge from HDKG, and decomposes the task into executable sub-goals. The action controller then sequentially executes these sub-goals. During execution, the Experience-Driven Reflector is activated periodically, leveraging historical experience from AMEP to assess whether Optimus-1 can complete the current sub-goal. If not, it instructs the Knowledge-Guided Planner to revise its plan. Through iterative interaction with the environment,Optimus-1 ultimately completes the task.
<img src="./assets/fig2.png" >

## :smile_cat: Evaluation results
We report the `average success rate (SR)`, `average number of steps (AS)`, and `average time (AT)` on each task group, the results of each task can be found in the Appendix experiment. Lower AS and AT metrics mean that the agent is more efficient at completing the task, while $∞$ indicates that the agent is unable to complete the task. Overall represents the average result on the five groups of Iron, Gold, Diamond, Redstone, and Armor.
<img src="./assets/table1.png" >

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@misc{li2024optimus1hybridmultimodalmemory,
      title={Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks}, 
      author={Zaijing Li and Yuquan Xie and Rui Shao and Gongwei Chen and Dongmei Jiang and Liqiang Nie},
      year={2024},
      eprint={2408.03615},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2408.03615}, 
}
```
