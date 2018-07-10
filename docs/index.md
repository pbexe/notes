# Markdown reference

## Standard stuff

*Standard* **markdown** stuff

```python
for i in range (1,100):
    print("Syntax highlighting")
    print("George was here")
```

## Notes etc.

!!! note
    This is a note

!!! bug
    This is a bug

!!! info
    These work for:

    - Note
    - Abstract
    - Info
    - Tip
    - Success
    - Question
    - Warning
    - Failure
    - Danger
    - Bug
    - Example
    - Quote

Clicky boi:

??? Question "What is the answer to life the universe and everything?"
    42

## Maths

$$
f(x) = \left\{
  \begin{array}{lr}
    x^2 & : x < 0\\
    x^3 & : x \ge 0
  \end{array}
\right.
$$

## Other stuff

- Emojis :smile:
- Inline highlighting for `#!python print("code")`
- Auto adding of https://www.google.co.uk/search?q=links
- Highlighting of ==important== stuff
- Useful arrows and stuff --> +/-
- Tasks:
    * [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
    * [x] Nulla lobortis egestas semper
    * [x] Curabitur elit nibh, euismod et ullamcorper at, iaculis feugiat est
    * [ ] Vestibulum convallis sit amet nisi a tincidunt
        * [x] In hac habitasse platea dictumst
        * [x] In scelerisque nibh non dolor mollis congue sed et metus
        * [x] Sed egestas felis quis elit dapibus, ac aliquet turpis mattis
        * [ ] Praesent sed risus massa
    * [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque
    * [ ] Nulla vel eros venenatis, imperdiet enim id, faucibus nisi


## Installation

Clone:
`git clone ...`

Install:
`pipenv install`

Live reload:
`pipenv run mkdocs serve`

Build:
`pipenv run mkdocs build`

Deploy:
`pipenv run mkdocs gh-deploy`