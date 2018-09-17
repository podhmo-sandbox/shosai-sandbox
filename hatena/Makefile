WRAP ?= python -m reqtrace
init:
	shosai hatena search --save
fetch:
	# shosai hatena search -q="https://pod.hatenablog.com/entry/2018/09/09/175358"
	${WRAP} shosai hatena search -q="10257846132625017003"
pull:
	${WRAP} shosai hatena pull  docs/entry/2018/09/09/175358.md
push:
	${WRAP} shosai hatena push --draft hello.md
push2:
	${WRAP} shosai hatena push --draft image.md
push3:
	rm -rf K_Nearest_Neighbor_Dynamic_Time_Warping*
	wget https://raw.githubusercontent.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
	jupyter-nbconvert --to markdown K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
	${WRAP} shosai hatena push --draft K_Nearest_Neighbor_Dynamic_Time_Warping.md

