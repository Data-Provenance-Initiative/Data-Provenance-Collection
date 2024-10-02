# LaTeX preamble

Add this to the preamble to support symbols and formatting used in tables
```
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{makecell}
\usepackage{adjustbox}
\usepackage{colortbl}

% include line-size pdfs rather than use emoji (to avoid lualatex) and tikz (for simplicity)
\newcommand{\emojirobot}{\includegraphics[height=1em]{emojis/robot.pdf}}
\newcommand{\emojiglobe}{\includegraphics[height=1em]{emojis/globe-with-meridians.pdf}}
\newcommand{\emojiblank}{\phantom{\includegraphics[height=1em]{emojis/robot.pdf}}}
\newcommand{\NCDataCircle}{\includegraphics[height=0.9em]{emojis/NCDataCircle.pdf}}
\newcommand{\UnspecifiedDataCircle}{\includegraphics[height=0.9em]{emojis/UnspecifiedDataCircle.pdf}}
\newcommand{\CommercialDataCircle}{\includegraphics[height=0.9em]{emojis/CommercialDataCircle.pdf}}
\newcommand{\TransparentCircle}{\phantom{\includegraphics[height=0.9em]{emojis/CommercialDataCircle.pdf}}}
\newcommand{\greencheck}{\includegraphics[height=0.9em]{emojis/greencheck.pdf}}
\newcommand{\redcross}{\includegraphics[height=0.9em]{emojis/redcross.pdf}}
```

# After paste into doc
Formatting tweaks still needed after pasting these tables into the LaTeX doc:
* Edit the subsequent-pages captions. Pandas defaults to the full caption, which is too long, and doesn't let you change it.