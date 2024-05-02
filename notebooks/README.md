# LaTeX preamble

Add this to the preamble to get symbols used in the tables:

```
\usepackage{emoji}
\usepackage{tikz}
\usepackage{makecell}
\usepackage{colortbl}
\usepackage{longtable}

\newcommand{\emojiblank}{\phantom{\emoji{smile}}}
\newcommand{\NCDataCircle}{\tikz[baseline=-0.85ex]{\definecolor{mycolor}{HTML}{e04c71} \fill[mycolor] (0,0) circle (0.85ex);}}
\newcommand{\UnspecifiedDataCircle}{\tikz[baseline=-0.85ex]{\definecolor{mycolor}{HTML}{e0cd92} \fill[mycolor] (0,0) circle (0.85ex);}}
\newcommand{\CommercialDataCircle}{\tikz[baseline=-0.85ex]{\definecolor{mycolor}{HTML}{82b5cf} \fill[mycolor] (0,0) circle (0.85ex);}}
\newcommand{\TransparentCircle}{\tikz[baseline=-0.85ex]{\fill[fill opacity=0] (0,0) circle (0.85ex);}}

\definecolor{darkerGreen}{RGB}{0,170,0}
\newcommand\greencheck{\textcolor{darkerGreen}{\ding{52}}}
\newcommand\redcross{\textcolor{red}{\ding{55}}}
\newcommand\orangecircle{\textcolor{orange}{\ding{108}}}
```
