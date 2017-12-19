# The Problem
You are a zookeeper in the reptile house. One of your rare lizards has just had several babies. Your job is to find a place to put each baby lizard in a nursery. However, there is a catch: the baby lizards have very long tongues.

A baby lizard can shoot out its tongue and eat any other baby lizard before you have time to save it. As such, you want to make sure that no baby lizard can eat another baby lizard in the nursery (burp).

For each baby lizard, you can place them in one spot on a grid. From there, they can shoot out their tongue up, down, left, right and diagonally as well. Their tongues are very long and can reach to the edge of the nursery from any location.

Figure 1 shows in what ways a baby lizard can eat another.

![Alt text](/../screenshots/screenshots/Figure1.png?raw=true)

In addition to baby lizards, your nursery may have some trees planted in it. Your lizards cannot shoot their tongues through the trees nor can you move a lizard into the same place as a tree.

As such, a tree will block any lizard from eating another lizard if it is in the path. Additionally, the tree will block you from moving the lizard to that location.

Figure 2 shows some different valid arrangements of lizards:

![Alt text](/../screenshots/screenshots/Figure2.png?raw=true)

Given an arrangement of trees, we need to output a new arrangement of lizards such that no baby lizard can eat another one. You cannot move any of the trees.
