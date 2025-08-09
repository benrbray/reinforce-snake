---
marp: true
math: katex
header: 'Introduction to Reinforcement Learning'
---

<style>
section.section  { background-color: #cfe8ff; }
section.kitt     { background-color: #F0E68C; }
section.livecode { background-color: #cdf08c; }
section.question { background-color: #e6e0ff; }
section.optional { background-color: #eeeeee; }
</style>

# Introduction to Reinforcement Learning

> Benjamin Bray @ 9 August 2025

----------
<!-- _class: section -->
<!-- footer: "Reinforcement Learning: Basics" -->

## Reinforcement Learning:  Basics

----------

## Reinforcement Learning:  Setting

A robot (**"agent"**) must survive in an unknown environment...

* It has **sensors** which collect **observations** about the **state** of the world
* It can take **actions** to move or interact with the environment
* It receives **rewards** or **penalties** based on its actions.

----------

## Reinforcement Learning:  Challenges

* probabilistic state transitions (tires slip, fall off cliff!)

----------

## Reinforcement Learning:  Notation

* The **state space** $\mathcal{S}$ is a set of possible _states_
* Each state $s \in mathcal{S}$ has a (possibly different) set of actions, $\mathrm{Actions}(s)$

----------

----------
<!-- _class: section -->
<!-- footer: "Q-Learning" -->

## Q-Learning

The **Q-Learning Algorithm** is applicable when:

** **Action Space:** Discrete
** **State Space:** Discrete
** **Observations:** Perfect Information (?)

----------

## Q-Learning:  Idea

What if we had a table

----------
<!-- _class: section -->
<!-- footer: "Deep Q-Learning" -->

## Deep Q-Learning

----------
<!-- _class:  section -->
<!-- footer: "Policy Gradients" -->

## Policy Gradients
