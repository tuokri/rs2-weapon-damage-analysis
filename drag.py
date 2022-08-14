def drag_g1(mach: float) -> float:
    if mach >= 1.60:
        if mach >= 2.70:
            if mach >= 3.60:
                if mach >= 5.00:
                    return 0.4988
                elif mach >= 4.80:
                    return 0.4990
                elif mach >= 4.60:
                    return 0.4992
                elif mach >= 4.40:
                    return 0.4995
                elif mach >= 4.20:
                    return 0.4998
                elif mach >= 4.00:
                    return 0.5006
                elif mach >= 3.90:
                    return 0.5010
                elif mach >= 3.80:
                    return 0.5016
                elif mach >= 3.70:
                    return 0.5022
                else:  # if mach >= 3.60:
                    return 0.5030
            else:
                if mach >= 3.50:
                    return 0.5040
                elif mach >= 3.40:
                    return 0.5054
                elif mach >= 3.30:
                    return 0.5067
                elif mach >= 3.20:
                    return 0.5084
                elif mach >= 3.10:
                    return 0.5105
                elif mach >= 3.00:
                    return 0.5133
                elif mach >= 2.90:
                    return 0.5168
                elif mach >= 2.80:
                    return 0.5211
                else:  # if mach >= 2.70:
                    return 0.5264
        else:
            if mach >= 2.10:
                if mach >= 2.60:
                    return 0.5325
                elif mach >= 2.50:
                    return 0.5397
                elif mach >= 2.45:
                    return 0.5438
                elif mach >= 2.40:
                    return 0.5481
                elif mach >= 2.35:
                    return 0.5527
                elif mach >= 2.30:
                    return 0.5577
                elif mach >= 2.25:
                    return 0.5630
                elif mach >= 2.20:
                    return 0.5685
                elif mach >= 2.15:
                    return 0.5743
                else:  # if mach >= 2.10:
                    return 0.5804
            else:
                if mach >= 2.05:
                    return 0.5867
                elif mach >= 2.00:
                    return 0.5934
                elif mach >= 1.95:
                    return 0.6003
                elif mach >= 1.90:
                    return 0.6072
                elif mach >= 1.85:
                    return 0.6141
                elif mach >= 1.80:
                    return 0.6210
                elif mach >= 1.75:
                    return 0.6280
                elif mach >= 1.70:
                    return 0.6347
                elif mach >= 1.65:
                    return 0.6413
                else:  # if mach >= 1.60:
                    return 0.6474
    else:
        if mach >= 0.875:
            if mach >= 1.125:
                if mach >= 1.55:
                    return 0.6528
                elif mach >= 1.50:
                    return 0.6573
                elif mach >= 1.45:
                    return 0.6607
                elif mach >= 1.40:
                    return 0.6625
                elif mach >= 1.35:
                    return 0.6621
                elif mach >= 1.30:
                    return 0.6589
                elif mach >= 1.25:
                    return 0.6518
                elif mach >= 1.20:
                    return 0.6393
                elif mach >= 1.15:
                    return 0.6191
                else:  # if mach >= 1.125:
                    return 0.6053
            else:
                if mach >= 1.10:
                    return 0.5883
                elif mach >= 1.075:
                    return 0.5677
                elif mach >= 1.05:
                    return 0.5427
                elif mach >= 1.025:
                    return 0.5136
                elif mach >= 1.0:
                    return 0.4805
                elif mach >= 0.975:
                    return 0.4448
                elif mach >= 0.95:
                    return 0.4084
                elif mach >= 0.925:
                    return 0.3734
                elif mach >= 0.90:
                    return 0.3415
                else:  # if mach >= 0.875:
                    return 0.3136
        else:
            if mach >= 0.50:
                if mach >= 0.85:
                    return 0.2901
                elif mach >= 0.825:
                    return 0.2706
                elif mach >= 0.80:
                    return 0.2546
                elif mach >= 0.775:
                    return 0.2417
                elif mach >= 0.75:
                    return 0.2313
                elif mach >= 0.725:
                    return 0.2230
                elif mach >= 0.70:
                    return 0.2165
                elif mach >= 0.60:
                    return 0.2034
                elif mach >= 0.55:
                    return 0.2020
                else:  # if mach >= 0.50:
                    return 0.2032
            else:
                if mach >= 0.45:
                    return 0.2061
                elif mach >= 0.40:
                    return 0.2104
                elif mach >= 0.35:
                    return 0.2155
                elif mach >= 0.30:
                    return 0.2214
                elif mach >= 0.25:
                    return 0.2278
                elif mach >= 0.20:
                    return 0.2344
                elif mach >= 0.15:
                    return 0.2413
                elif mach >= 0.10:
                    return 0.2487
                elif mach >= 0.05:
                    return 0.2558
                else:
                    return 0.2629


def drag_g7(mach: float) -> float:
    if mach >= 1.60:
        if mach >= 2.70:
            if mach >= 3.60:
                if mach >= 5.00:
                    return 0.1618
                elif mach >= 4.80:
                    return 0.1672
                elif mach >= 4.60:
                    return 0.1730
                elif mach >= 4.40:
                    return 0.1793
                elif mach >= 4.20:
                    return 0.1861
                elif mach >= 4.00:
                    return 0.1935
                elif mach >= 3.90:
                    return 0.1975
                elif mach >= 3.80:
                    return 0.2017
                elif mach >= 3.70:
                    return 0.2060
                else:  # if mach >= 3.60:
                    return 0.2106
            else:
                if mach >= 3.50:
                    return 0.2154
                elif mach >= 3.40:
                    return 0.2205
                elif mach >= 3.30:
                    return 0.2258
                elif mach >= 3.20:
                    return 0.2313
                elif mach >= 3.10:
                    return 0.2368
                elif mach >= 3.00:
                    return 0.2424
                elif mach >= 2.95:
                    return 0.2451
                elif mach >= 2.90:
                    return 0.2479
                elif mach >= 2.85:
                    return 0.2506
                elif mach >= 2.80:
                    return 0.2533
                elif mach >= 2.75:
                    return 0.2561
                else:  # if mach >= 2.70:
                    return 0.2588
        else:
            if mach >= 2.10:
                if mach >= 2.65:
                    return 0.2615
                elif mach >= 2.60:
                    return 0.2643
                elif mach >= 2.55:
                    return 0.2670
                elif mach >= 2.50:
                    return 0.2697
                elif mach >= 2.45:
                    return 0.2725
                elif mach >= 2.40:
                    return 0.2752
                elif mach >= 2.35:
                    return 0.2779
                elif mach >= 2.30:
                    return 0.2807
                elif mach >= 2.25:
                    return 0.2835
                elif mach >= 2.20:
                    return 0.2864
                elif mach >= 2.15:
                    return 0.2892
                else:  # if mach >= 2.10:
                    return 0.2922
            else:
                if mach >= 2.05:
                    return 0.2951
                elif mach >= 2.00:
                    return 0.2980
                elif mach >= 1.95:
                    return 0.3010
                elif mach >= 1.90:
                    return 0.3042
                elif mach >= 1.85:
                    return 0.3078
                elif mach >= 1.80:
                    return 0.3117
                elif mach >= 1.75:
                    return 0.3160
                elif mach >= 1.70:
                    return 0.3209
                elif mach >= 1.65:
                    return 0.3260
                else:  # if mach >= 1.60:
                    return 0.3315
    else:
        if mach >= 0.875:
            if mach >= 1.125:
                if mach >= 1.55:
                    return 0.3376
                elif mach >= 1.50:
                    return 0.3440
                elif mach >= 1.40:
                    return 0.3580
                elif mach >= 1.35:
                    return 0.3657
                elif mach >= 1.30:
                    return 0.3732
                elif mach >= 1.25:
                    return 0.3810
                elif mach >= 1.20:
                    return 0.3884
                elif mach >= 1.15:
                    return 0.3955
                else:  # if mach >= 1.125:
                    return 0.3987
            else:
                if mach >= 1.10:
                    return 0.4014
                elif mach >= 1.075:
                    return 0.4034
                elif mach >= 1.05:
                    return 0.4043
                elif mach >= 1.025:
                    return 0.4015
                elif mach >= 1.0:
                    return 0.3803
                elif mach >= 0.975:
                    return 0.2993
                elif mach >= 0.95:
                    return 0.2054
                elif mach >= 0.925:
                    return 0.1660
                elif mach >= 0.90:
                    return 0.1464
                else:  # if mach >= 0.875:
                    return 0.1368
        else:
            if mach >= 0.50:
                if mach >= 0.85:
                    return 0.1306
                elif mach >= 0.825:
                    return 0.1266
                elif mach >= 0.80:
                    return 0.1242
                elif mach >= 0.775:
                    return 0.1226
                elif mach >= 0.75:
                    return 0.1215
                elif mach >= 0.725:
                    return 0.1207
                elif mach >= 0.70:
                    return 0.1202
                elif mach >= 0.65:
                    return 0.1197
                elif mach >= 0.60:
                    return 0.1194
                elif mach >= 0.55:
                    return 0.1193
                else:  # if mach >= 0.50:
                    return 0.1194
            else:
                if mach >= 0.45:
                    return 0.1193
                elif mach >= 0.40:
                    return 0.1193
                elif mach >= 0.35:
                    return 0.1194
                elif mach >= 0.30:
                    return 0.1194
                elif mach >= 0.25:
                    return 0.1194
                elif mach >= 0.20:
                    return 0.1193
                elif mach >= 0.15:
                    return 0.1194
                elif mach >= 0.10:
                    return 0.1196
                elif mach >= 0.05:
                    return 0.1197
                else:  # if mach >= 0.00:
                    return 0.1198
