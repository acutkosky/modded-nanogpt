# Update Smoothing (includes #124 and #128)


This is an update to #129 that incorporates #128 (Snoo optimizer).

TL;DR: iterations decrease from 5640 to 5600 after applying an EMA to the muon update.

Overall change is to apply a small EMA filter to the Muon update like so:
```
muon_update = NS(EMA(gradients))
final_update = EMA(muon_update)
```

The EMA weight is small (0.5, decaying to 0.2 over training). Additionally, the learning rate is tuned:
* Final learning rate is 0.01 * peak learning rate
* Muon LR increased to 0.03


Time improvement on my system:
#128 baseline = 23.68
With update smoothing/LR changes = 23.47
So overall 23.46/23.68 = 99.1%

My baseline times are consistently slower than the claimed runtimes. I suspect this is an issue with my cluster setup. I have tried a variety of package versions, and while the precise timings I see vary, the relative ordering of different commits does not.

# Statistics

For the baseline #129, I report the stats from the first 80 out of the 160 total runs performed in #129:
```
Processed 80 files.
val loss:
mean: 	2.919790
std:  	0.000824
median:	2.9196765
min:   	2.917344
max:   	2.921742
val loss 99% confidence interval: (2.919546 - 2.920033)
val_loss t-test p=0.012530 (small means <2.92)
train time (minutes): mean=23.5067, std=0.1826
train time 99% confidence interval: (23.4528 - 23.5606)
```
This one missed the p-value with this sample; with more samples it sufficed.

For the baseline #128, I ran 80 runs, with the following statistics:
```
Processed 80 files.
val loss:
mean: 	2.919758
std:  	0.000892
median:	2.9198180000000002
min:   	2.917598
max:   	2.922106
val loss 99% confidence interval: (2.919495 - 2.920021)
val_loss t-test p=0.008817 (small means <2.92)
train time (minutes): mean=23.6758, std=0.1655
train time 99% confidence interval: (23.6270 - 23.7246)
```
This hits the required p-value, but I did find that I could not hit this value reliably.
In fact, while debugging, I ran several rounds of this 80 runs experiment, and only most of them did not achieve a p-value < 0.01. For other commits (e.g. #124, #119), I see p-values in line with the reported values, so I'm guessing there is just a problem with variance in p-values for these optimization changes for some reason.


For the update smoothing change, I ran 2 replicates of 40 runs each to ensure that both were below the p-value target.
```
Replicate 1:
Processed 40 files.
val loss:
mean: 	2.919503
std:  	0.000946
median:	2.919372
min:   	2.917394
max:   	2.92163
val loss 99% confidence interval: (2.919098 - 2.919907)
val_loss t-test p=0.000968 (small means <2.92)
train time (minutes): mean=23.4627, std=0.1625
train time 99% confidence interval: (23.3932 - 23.5322)

Replicate 2:
Processed 40 files.
val loss:
mean: 	2.919646
std:  	0.000832
median:	2.9193854999999997
min:   	2.918298
max:   	2.921653
val loss 99% confidence interval: (2.919290 - 2.920002)
val_loss t-test p=0.005214 (small means <2.92)
train time (minutes): mean=23.4693, std=0.1741
train time 99% confidence interval: (23.3948 - 23.5438)
```

The overall stats for 80 runs:
```
Processed 80 files.
val loss:
mean: 	2.919574
std:  	0.000888
median:	2.9193854999999997
min:   	2.917394
max:   	2.921653
val loss 99% confidence interval: (2.919312 - 2.919836)
val_loss t-test p=0.000025 (small means <2.92)
train time (minutes): mean=23.4660, std=0.1674
train time 99% confidence interval: (23.4166 - 23.5154)
```


Note that for a more stringent p-value variance bound, increasing the iteration count to 5610 iterations yielded 4 groups of 20 runs all of which had p-values less than 0.01. However, this method was no faster than #129.

To be very safe about avoiding this p-value variance issue and keep the number of runs
required to verify manageable, I increased the number of iterations to 5610 and ran 4 replicates of 20 runs each (80 total runs).
Each replicate was required to have a p-value < 0.01.
This method is slightly slower than than #129, but still faster than #128: the average time was 23.5451 minutes (which should be expected; the number of tierations is the same but now there is a small Snoo overhead).
However, the p-value seems very robust.
```
Replicate 1:
Processed 20 files.
val loss:
mean: 	2.919040
std:  	0.000932
median:	2.919213
min:   	2.91748
max:   	2.920562
val loss 99% confidence interval: (2.918446 - 2.919633)
val_loss t-test p=0.000096 (small means <2.92)
train time (minutes): mean=23.5332, std=0.1660
train time 99% confidence interval: (23.4276 - 23.6388)

Replicate 2:
Processed 20 files.
val loss:
mean: 	2.919403
std:  	0.000897
median:	2.9191345
min:   	2.917866
max:   	2.920879
val loss 99% confidence interval: (2.918832 - 2.919974)
val_loss t-test p=0.003883 (small means <2.92)
train time (minutes): mean=23.5423, std=0.1737
train time 99% confidence interval: (23.4317 - 23.6528)

Replicate 3:
Processed 20 files.
val loss:
mean: 	2.919411
std:  	0.000737
median:	2.9194455
min:   	2.918034
max:   	2.921031
val loss 99% confidence interval: (2.918942 - 2.919879)
val_loss t-test p=0.001004 (small means <2.92)
train time (minutes): mean=23.5387, std=0.1499
train time 99% confidence interval: (23.4433 - 23.6341)

Replicate 4:
Processed 20 files.
val loss:
mean: 	2.919277
std:  	0.000768
median:	2.9192204999999998
min:   	2.918204
max:   	2.921505
val loss 99% confidence interval: (2.918788 - 2.919766)
val_loss t-test p=0.000239 (small means <2.92)
train time (minutes): mean=23.5661, std=0.1492
train time 99% confidence interval: (23.4712 - 23.6610)
```

# Raw data

Here are 40 validation losses:
```
2.920179
2.918443
2.919242
2.921438
2.920819
2.920151
2.920055
2.919716
2.919854
2.920454
2.920188
2.918091
2.919119
2.918759
2.918891
2.918227
2.917394
2.918941
2.920784
2.918777
2.919333
2.920715
2.918615
2.919949
2.921143
2.92163
2.918061
2.919139
2.919949
2.919411
2.919704
2.918636
2.918776
2.919722
2.919165
2.919273
2.919465
2.919575
2.919214
2.919108
```

and 40 timing values:
```
1406423
1401191
1411527
1434813
1399497
1399971
1417149
1389741
1400306
1410344
1408796
1400728
1420049
1411749
1412874
1409844
1403704
1405557
1396725
1399304
1399873
1411844
1427380
1423732
1393361
1416559
1398423
1403159
1418918
1395124
1398572
1413117
1412057
1414488
1417521
1412134
1398779
1404215
1410816
1400153
```
