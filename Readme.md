#### Data Mining using Decision Trees

- Implemented decision trees from scratch using numpy and pandas
- Two types of decision trees implemented :-
	- Vanilla - Simple division of nodes based on Information Gain
	- Prune - Nodes are pruned to prevent overfitting based on validation set
	
To Run :-
	- python3 ID3.py train.csv test.csv prune `<train-set-percent>` `<validation-set-percent>`
	- python3 ID3.py train.csv test.csv vanilla `<train-set-percent>`
	- python3 plots_prune.py
	- python3 plots_without_prune.py
