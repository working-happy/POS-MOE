# POS-MOE


Author

Jianying Chen

gochenjianying@m.scnu.edu.cn

June 11, 2021

## Data
- Step 1: Download each tweet's dataset include text and images via this link

​       (https://drive.google.com/drive/folders/19mMZufakzp96tOciqfdwknL0LXSv3dIV?usp=sharing)

- Step 2: Download the pre-trained Mask-RCNN via this link

​       (https://drive.google.com/file/d/15GfqpG8ertVLMf0wNA3NslOX1-9r-NG1/view?usp=sharing)

- Step 3: Put the dataset under the folder named "data" and the Mask-RCNN model under the folder named "object_detector"
- Step 4: Change the dataset path in line 610 and line 615 of the "run_mtmner_crf.py" file
- Step 5: Change the object detector path in line 76 of the "detector.py" file

## Requirement

* PyTorch 1.0.0
* Python 3.7

## Code Usage

### Training for UMT
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES=2" based on your available GPUs.

```sh
sh run_mtmner_crf.sh
```

## Result

- We show our running logs on twitter-2015 and twitter-2017 in the folder "log files". 

<table>
	<tr>
        <td>Methods</td>
        <td>PER(F1)</td>
		<td>LOC(F1)</td>
		<td>ORG(F1)</td>
		<td>MISC(F1)</td>
		<td>P(ALL)</td>
		<td>R(ALL)</td>
		<td>F1(ALL)</td>
    </tr>
    <tr>
		<td>twitter2015</td>
        <td>83.98</td>
		<td>81.55</td>
		<td>63.07</td>
		<td>43.89</td>
		<td>72.82</td>
		<td>75.76</td>
		<td>74.26</td>
    </tr>
    <tr>
		<td>twitter2017</td>
        <td>91.94</td>
		<td>85.96</td>
		<td>83.48</td>
		<td>69.06</td>
		<td>86.64</td>
		<td>85.42</td>
		<td>86.02</td>
    </tr>

</table>

