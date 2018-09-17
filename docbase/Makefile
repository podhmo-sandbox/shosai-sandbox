WRAP ?= python -m reqtrace

search:
	${WRAP} shosai docbase search
clonehello:
	${WRAP} shosai docbase clone https://po.docbase.io/posts/528390 --name hello
pullhello:
	${WRAP} shosai docbase pull docs/hello.md
pushhello:
	${WRAP} shosai docbase push docs/hello.md
pushimage:
	${WRAP} shosai docbase push image.md
pushipynb:
	jupyter-nbconvert --to markdown example.ipynb
	${WRAP} shosai docbase push example.md
pushipynb2:
	rm -r K_Nearest_Neighbor_Dynamic_Time_Warping*
	wget https://raw.githubusercontent.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
	jupyter-nbconvert --to markdown K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
	${WRAP} shosai docbase push K_Nearest_Neighbor_Dynamic_Time_Warping.md
