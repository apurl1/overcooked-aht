# Overcooked AHT

This repository has code based on PantheonRL, Overcooked AI, and HSP to train PPO (from StableBaselines3) with diverse partner agents.

## Installation
```
# Clone Overcooked AHT
git clone https://github.com/apurl1/overcooked-aht.git
cd overcooked-aht

# Create conda environment
conda env create -f environment.yml
conda activate overcookedaht
```

## PantheonRL

The PantheonRL directory is adapted from this [repository](https://github.com/Stanford-ILIAD/PantheonRL).

PantheonRL is a package for training and testing multi-agent reinforcement learning environments. The goal of PantheonRL is to provide a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more. PantheonRL also provides a web user interface suitable for lightweight experimentation and prototyping.

PantheonRL is built on top of StableBaselines3 (SB3), allowing direct access to many of SB3's standard RL training algorithms such as PPO. PantheonRL currently follows a decentralized training paradigm -- each agent is equipped with its own replay buffer and update algorithm. The agents objects are designed to be easily manipulable. They can be saved, loaded and plugged into different training procedures such as self-play, ad-hoc / cross-play, round-robin training, or finetuning.

```
"PantheonRL: A MARL Library for Dynamic Training Interactions"
Bidipta Sarkar*, Aditi Talati*, Andy Shih*, Dorsa Sadigh
In Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track), 2022

@inproceedings{sarkar2021pantheonRL,
  title={PantheonRL: A MARL Library for Dynamic Training Interactions},
  author={Sarkar, Bidipta and Talati, Aditi and Shih, Andy and Sadigh Dorsa},
  booktitle = {Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track)},
  year={2022}
}
```

## Overcooked-AI

The Overcooked game environment is taken from this [repository](https://github.com/HumanCompatibleAI/overcooked_ai).

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

```
"On the Utility of Learning about Humans for Human-AI Coordination"
Micah Carroll, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, Anca Dragan
In Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)

@inproceedings{carroll2019overcooked,
  title={On the Utility of Learning about Humans for Human-AI Coordination},
  author={Carroll, Micah and Rohin Shah and Mark K. Ho and Thomas L. Griffiths and Sanjit A. Seshia and Pieter Abbeel and Anca Dragan},
  booktitle = {33rd Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## Script Agents

The script agents are taken from the Hidden-utility Self Play repository [HSP](https://github.com/samjia2000/HSP/tree/main):

```
@inproceedings{
yu2023learning,
title={Learning Zero-Shot Cooperation with Humans, Assuming Humans Are Biased},
author={Chao Yu and Jiaxuan Gao and Weilin Liu and Botian Xu and Hao Tang and Jiaqi Yang and Yu Wang and Yi Wu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=TrwE8l9aJzs}
}
```
