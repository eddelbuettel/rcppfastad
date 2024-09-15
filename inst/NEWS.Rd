\name{NEWS}
\title{News for Package \pkg{RcppFastAD}}
\newcommand{\ghpr}{\href{https://github.com/eddelbuettel/rcppfastad/pull/#1}{##1}}
\newcommand{\ghit}{\href{https://github.com/eddelbuettel/rcppfastad/issues/#1}{##1}}

\section{Changes in version 0.0.3 (2024-09-15)}{
  \itemize{
    \item The package now compiles under the C++20 standard to avoid a warning
    under \code{clang++-18} (Dirk addressing \ghit{9})
    \item Minor updates to continuous integration and badges have been made
    as well
  }
}

\section{Changes in version 0.0.2 (2023-03-05)}{
  \itemize{
    \item One C++ operation is protected from operating on a
    \code{nullptr}
    \item Additional tests have been added, tests now cover all three
    demo / example functions
    \item Return values and code for the examples
    \code{linear_regression} and \code{quadratic_expression} have been
    adjusted 
  }
}

\section{Changes in version 0.0.1 (2023-02-24)}{
  \itemize{
    \item Initial release version and CRAN upload
  }
}

