## Results
#### ChatGPT
| Accuracy | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	     |---	    |---  	|---	  |---	  |---  	|
| BS=1     |  86.8%	|     	|     	|     	|    	 |
| BS=16    |  77.5% |  80%  |79.7% | 82.5%  |  81.3%  |
| BS=32    |  70%   | 75.9% |77.2% |77.8%  |  77.8%  |

| Weighted (Conf) | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	     |---	    |---  	|---	  |---	  |---  	|
| BS=1     |  86.9%|     	|     	|     	|    	 |
| BS=16    |  76.9%  | 82.2% |81.6% | 82.2%  |  81.3%  |
| BS=32    |  71.9%    | 77.8% |79.4% |78.4%  |  78.4%  |

| Token Num | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	      |---	  |---  	|---	  |---	  |---  	|
| BS=1 	    |      	|   - 	|   -  	|   - 	|   -   |
| BS=16     |      	|     	|     	|     	|    	  |
| BS=32     |      	|     	|     	|     	|    	  |


| Calling Num | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	        |---	  |---  	|---	  |---	  |---  	|
| BS=1 	      |      	|   - 	|   -  	|   - 	|   -   |
| BS=16       |      	|     	|     	|     	|    	  |
| BS=32       |      	|     	|     	|     	|    	  |


#### GPT4
| Accuracy | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	     |---	    |---  	|---	  |---	  |---  	|
| BS=1 	   |  90.6%    	|   - 	|   -  	|   - 	|   -   |
| BS=16    | 89.1%| 89.4%|89.1%| 89.4%|89.5%
| BS=32    | 87.8%  | 89.7% |    90.6% 	|   90.6%| 90.9%  |
| BS=64    | 72.8%   | 76.9%|   82.8%	|  85.3%	|  86.3%  |

| Weighted (Conf) | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	     |---	    |---  	|---	  |---	  |---  	|
| BS=1 	   |  90.9%   	|   - 	|   -  	|   - 	|   -   |
| BS=16    | 89.7%| 90% |90%| 90%|90.6%
| BS=32    | 87.2%  | 89.7% |    90.3% 	|   90%| 89.7%  |
| BS=64    | 75.3%   | 82.8%|   80.9%	|  84.4%	|  84.4%  |

| Token Num | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	      |---	  |---  	|---	  |---	  |---  	|
| BS=1 	    |      	|   - 	|   -  	|   - 	|   -   |
| BS=16     |      	|     	|     	|     	|    	  |
| BS=32     |      	|     	|     	|     	|    	  |
| BS=64     |      	|     	|     	|     	|    	  |

| Calling Num | VT=1 	| VT=3 	| VT=5 	| VT=7 	| VT=9 	|
|---	        |---	  |---  	|---	  |---	  |---  	|
| BS=1 	      |      	|   - 	|   -  	|   - 	|   -   |
| BS=16       |      	|     	|     	|     	|    	  |
| BS=32       |      	|     	|     	|     	|    	  |
| BS=64       |      	|     	|     	|     	|    	  |