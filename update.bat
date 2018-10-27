git add --all
@echo off
set/p commitInfo = info>
git commit -m @commitInfo@
git push origin master
