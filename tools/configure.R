
if (dir.exists(".git")) {
    ## development from a .git directory can use these flags
    xtraflags <- "-Wno-ignored-attributes -Wno-maybe-uninitialized"
} else {
    ## else build from tarball so stick with existing flags
    xtraflags <- ""
}
win <- if (Sys.info()[["sysname"]] == "Windows") ".win" else ""
infile <- file.path("src", paste0("Makevars", win, ".in"))
outfile <- file.path("src", paste0("Makevars", win))
lines <- readLines(infile)
lines <- gsub("@XTRAFLAGS@", xtraflags, lines)
writeLines(lines, outfile)
