import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--in_csv_file", default="train_labels.csv", type=str)
parser.add_argument("--out_csv_file", default="train.csv", type=str)
args = parser.parse_args()

train = pd.read_csv(f"../../RSNA-BTC-Datasets/{args.in_csv_file}")

targets = []
target = "MGMT_value"

if args.in_csv_file == "upenn_train_labels.csv":
    train_idxs = [33, 190, 245, 129, 253, 21, 237, 146, 16, 204, 75, 173, 135, 209, 179, 4, 96, 188, 61, 67, 52, 66, 26, 110, 233, 124, 268, 40, 13, 107, 145, 166, 3, 159, 24, 30, 252, 213, 60, 56, 248, 157, 46, 19, 212, 153, 54, 227, 80, 51, 2, 196, 104, 150, 86, 10, 168, 58, 41, 14, 50, 194, 123, 62, 158, 187, 130, 246, 154, 43, 221, 138, 254, 181, 149, 112, 207, 98, 180, 93, 171, 162, 36, 113, 0, 94, 95, 223, 69, 49, 48, 85, 270, 141, 23, 249, 143, 78, 100, 131, 228, 271, 6, 68, 84, 170, 121, 140, 214, 240, 234, 239, 91, 241, 257, 11, 119, 102, 35, 57, 169, 65, 1, 120, 226, 186, 42, 105, 132, 244, 17, 38, 133, 53, 164, 217, 128, 34, 28, 183, 114, 203, 163, 151, 202, 31, 232, 127, 185, 250, 260, 32, 167, 142, 236, 147, 29, 177, 216, 99, 82, 269, 175, 79, 197, 243, 208, 115, 148, 266, 72, 77, 25, 165, 81, 261, 174, 263, 39, 193, 88, 70, 87, 242, 211, 9, 195, 251, 192, 117, 47, 172, 413, 531, 485, 274, 430, 285, 426, 404, 451, 455, 349, 453, 308, 375, 416, 480, 425, 382, 304, 504, 524, 513, 435, 283, 320, 488, 496, 479, 462, 338, 315, 276, 385, 406, 481, 309, 465, 460, 407, 282, 386, 448, 280, 391, 359, 444, 392, 475, 388, 541, 374, 370, 469, 327, 303, 439, 322, 373, 478, 499, 542, 329, 333, 279, 281, 352, 467, 540, 432, 507, 311, 438, 394, 483, 535, 502, 442, 397, 427, 353, 332, 500, 321, 326, 343, 484, 519, 328, 509, 431, 544, 307, 390, 534, 510, 495, 325, 347, 299, 446, 525, 440, 277, 412, 454, 538, 273, 278, 414, 334, 408, 523, 395, 295, 341, 293, 511, 287, 443, 305, 543, 501, 476, 337, 456, 498, 393, 354, 331, 286, 489, 527, 345, 420, 433, 360, 396, 520, 532, 457, 409, 324, 468, 464, 389, 297, 367, 437, 379, 289, 336, 487, 401, 459, 378, 482, 508, 301, 346, 492, 497, 515, 362, 474, 422, 447, 366, 314, 450, 356, 316, 365, 291, 472, 471, 418, 423, 312, 364, 384, 526, 317, 342, 275, 494, 419, 514, 302, 387, 298, 449, 441]
    
    val_idxs = [218, 259, 225, 111, 178, 161, 89, 220, 55, 198, 265, 267, 210, 126, 97, 15, 8, 152, 59, 238, 45, 116, 247, 160, 63, 7, 92, 235, 206, 255, 144, 118, 103, 191, 73, 137, 230, 83, 64, 258, 101, 200, 272, 106, 27, 224, 189, 182, 219, 37, 231, 136, 20, 76, 318, 284, 402, 330, 357, 411, 503, 410, 372, 344, 323, 436, 505, 351, 533, 400, 398, 491, 376, 355, 415, 516, 363, 545, 521, 434, 539, 428, 288, 466, 518, 371, 537, 290, 335, 403, 493, 340, 350, 522, 528, 405, 429, 473, 296, 445, 377, 517, 424, 348, 477, 319, 463, 529]
    
    test_idxs = [201, 229, 134, 176, 5, 22, 199, 184, 12, 122, 215, 74, 44, 222, 125, 156, 155, 108, 18, 139, 256, 90, 71, 109, 205, 262, 264, 486, 310, 294, 383, 292, 369, 421, 361, 300, 461, 458, 339, 490, 512, 506, 399, 452, 380, 470, 306, 368, 313, 530, 417, 536, 381, 358]
    
    for i in range(len(train)):
        if i in train_idxs:
            train.loc[i, "split"] = 0
        elif i in val_idxs:
            train.loc[i, "split"] = 1
        elif i in test_idxs:
            train.loc[i, "split"] = 2
    
elif args.out_csv_file == "train.csv":
    train_idxs = [238, 137, 106, 284, 44, 139, 247, 288, 156, 297, 252, 54, 234, 18, 205, 254, 182, 56, 71, 144, 249, 209, 290, 219, 158, 176, 33, 83, 136, 210, 118, 60, 159, 282, 110, 21, 29, 150, 16, 75, 109, 179, 283, 4, 96, 229, 61, 67, 295, 266, 171, 281, 40, 189, 13, 107, 200, 3, 161, 125, 24, 30, 77, 279, 190, 19, 257, 235, 268, 80, 51, 2, 239, 104, 262, 86, 10, 224, 58, 41, 14, 155, 50, 215, 237, 123, 220, 62, 191, 230, 130, 213, 187, 43, 114, 138, 199, 222, 149, 112, 298, 98, 221, 93, 208, 162, 36, 178, 113, 0, 94, 294, 95, 299, 263, 256, 69, 49, 48, 85, 300, 141, 207, 23, 250, 148, 143, 78, 180, 100, 204, 131, 269, 301, 196, 6, 68, 203, 84, 170, 121, 140, 258, 276, 142, 259, 91, 82, 285, 11, 119, 102, 35, 57, 169, 231, 65, 1, 120, 267, 186, 42, 105, 132, 79, 17, 271, 38, 53, 260, 128, 28, 183, 163, 151, 244, 202, 31, 32, 127, 185, 280, 273, 147, 278, 177, 99, 197, 243, 115, 265, 72, 25, 165, 289, 174, 291, 39, 193, 88, 70, 87, 292, 242, 277, 211, 9, 195, 251, 192, 117, 47, 172, 516, 380, 549, 365, 554, 536, 489, 433, 486, 369, 338, 320, 465, 589, 448, 528, 499, 348, 353, 360, 556, 547, 531, 544, 335, 381, 449, 436, 399, 517, 401, 505, 471, 392, 543, 561, 550, 435, 349, 500, 410, 473, 303, 542, 374, 579, 405, 370, 411, 525, 387, 584, 557, 428, 539, 412, 329, 484, 602, 545, 490, 437, 372, 402, 527, 521, 342, 496, 493, 432, 551, 443, 434, 564, 416, 321, 379, 495, 552, 414, 479, 312, 492, 462, 599, 346, 394, 407, 441, 478, 415, 371, 339, 513, 368, 345, 429, 306, 420, 451, 596, 580, 574, 431, 485, 440, 600, 310, 389, 482, 422, 569, 418, 464, 332, 458, 563, 488, 404, 400, 323, 456, 357, 333, 583, 352, 403, 565, 594, 454, 417, 359, 447, 363, 497, 535, 515, 502, 601, 309, 562, 311, 520, 382, 576, 511, 577, 546, 568, 470, 341, 373, 424, 582, 347, 603, 597, 480, 361, 466, 427, 457, 383, 362, 595, 351, 356, 461, 354, 575, 522, 377, 307, 442, 510, 590, 559, 308, 444, 364, 438, 386, 530, 325, 573, 523, 506, 423, 384, 316, 587, 375, 450, 463, 390, 541, 605, 439, 498, 494, 419, 327, 397, 467, 409, 319, 366, 408, 512, 331, 504, 477, 396]

    val_idxs = [225, 152, 228, 201, 52, 245, 175, 168, 223, 217, 111, 135, 218, 12, 15, 66, 97, 90, 198, 103, 22, 212, 226, 264, 133, 216, 275, 270, 154, 55, 194, 255, 134, 8, 157, 241, 240, 81, 214, 167, 5, 59, 92, 274, 34, 145, 116, 188, 246, 7, 45, 129, 122, 63, 124, 227, 146, 302, 26, 108, 560, 472, 367, 518, 487, 343, 514, 459, 588, 426, 524, 355, 538, 566, 395, 474, 425, 468, 558, 324, 567, 326, 591, 328, 592, 571, 337, 322, 314, 391, 421, 586, 453, 455, 508, 570, 491, 406, 501, 388, 378, 534, 452, 598, 385, 578, 315, 336, 581, 555, 413, 507, 334, 445, 529, 330, 340, 317, 604, 481]

    test_idxs = [89, 74, 153, 64, 296, 287, 286, 236, 126, 73, 20, 46, 160, 232, 181, 27, 173, 261, 37, 101, 166, 233, 184, 164, 206, 248, 253, 293, 76, 272, 305, 585, 304, 475, 476, 533, 548, 553, 344, 318, 393, 483, 532, 509, 540, 313, 430, 350, 526, 572, 460, 446, 398, 469, 537, 593, 358, 376, 519, 503]
    
    for i in range(len(train)):
        if i in train_idxs:
            train.loc[i, "split"] = 0
        elif i in val_idxs:
            train.loc[i, "split"] = 1
        elif i in test_idxs:
            train.loc[i, "split"] = 2
elif args.out_csv_file == "train_f.csv":
    train_idxs = [33, 190, 245, 129, 253, 21, 237, 146, 16, 204, 75, 173, 135, 209, 179, 4, 96, 188, 61, 67, 52, 66, 26, 110, 233, 124, 268, 40, 13, 107, 145, 166, 3, 159, 24, 30, 252, 213, 60, 56, 248, 157, 46, 19, 212, 153, 54, 227, 80, 51, 2, 196, 104, 150, 86, 10, 168, 58, 41, 14, 50, 194, 123, 62, 158, 187, 130, 246, 154, 43, 221, 138, 254, 181, 149, 112, 207, 98, 180, 93, 171, 162, 36, 113, 0, 94, 95, 223, 69, 49, 48, 85, 270, 141, 23, 249, 143, 78, 100, 131, 228, 271, 6, 68, 84, 170, 121, 140, 214, 240, 234, 239, 91, 241, 257, 11, 119, 102, 35, 57, 169, 65, 1, 120, 226, 186, 42, 105, 132, 244, 17, 38, 133, 53, 164, 217, 128, 34, 28, 183, 114, 203, 163, 151, 202, 31, 232, 127, 185, 250, 260, 32, 167, 142, 236, 147, 29, 177, 216, 99, 82, 269, 175, 79, 197, 243, 208, 115, 148, 266, 72, 77, 25, 165, 81, 261, 174, 263, 39, 193, 88, 70, 87, 242, 211, 9, 195, 251, 192, 117, 47, 172, 413, 531, 485, 274, 430, 285, 426, 404, 451, 455, 349, 453, 308, 375, 416, 480, 425, 382, 304, 504, 524, 513, 435, 283, 320, 488, 496, 479, 462, 338, 315, 276, 385, 406, 481, 309, 465, 460, 407, 282, 386, 448, 280, 391, 359, 444, 392, 475, 388, 541, 374, 370, 469, 327, 303, 439, 322, 373, 478, 499, 542, 329, 333, 279, 281, 352, 467, 540, 432, 507, 311, 438, 394, 483, 535, 502, 442, 397, 427, 353, 332, 500, 321, 326, 343, 484, 519, 328, 509, 431, 544, 307, 390, 534, 510, 495, 325, 347, 299, 446, 525, 440, 277, 412, 454, 538, 273, 278, 414, 334, 408, 523, 395, 295, 341, 293, 511, 287, 443, 305, 543, 501, 476, 337, 456, 498, 393, 354, 331, 286, 489, 527, 345, 420, 433, 360, 396, 520, 532, 457, 409, 324, 468, 464, 389, 297, 367, 437, 379, 289, 336, 487, 401, 459, 378, 482, 508, 301, 346, 492, 497, 515, 362, 474, 422, 447, 366, 314, 450, 356, 316, 365, 291, 472, 471, 418, 423, 312, 364, 384, 526, 317, 342, 275, 494, 419, 514, 302, 387, 298, 449, 441]
    
    val_idxs = [218, 259, 225, 111, 178, 161, 89, 220, 55, 198, 265, 267, 210, 126, 97, 15, 8, 152, 59, 238, 45, 116, 247, 160, 63, 7, 92, 235, 206, 255, 144, 118, 103, 191, 73, 137, 230, 83, 64, 258, 101, 200, 272, 106, 27, 224, 189, 182, 219, 37, 231, 136, 20, 76, 318, 284, 402, 330, 357, 411, 503, 410, 372, 344, 323, 436, 505, 351, 533, 400, 398, 491, 376, 355, 415, 516, 363, 545, 521, 434, 539, 428, 288, 466, 518, 371, 537, 290, 335, 403, 493, 340, 350, 522, 528, 405, 429, 473, 296, 445, 377, 517, 424, 348, 477, 319, 463, 529]
    
    test_idxs = [201, 229, 134, 176, 5, 22, 199, 184, 12, 122, 215, 74, 44, 222, 125, 156, 155, 108, 18, 139, 256, 90, 71, 109, 205, 262, 264, 486, 310, 294, 383, 292, 369, 421, 361, 300, 461, 458, 339, 490, 512, 506, 399, 452, 380, 470, 306, 368, 313, 530, 417, 536, 381, 358]
    
    f0_list = ['00018', '00019', '00021', '00022', '00030', '00036', '00053', '00064', '00072', '00088', '00090', '00095', '00097', '00099', '00116', '00121', '00122', '00124', '00130', '00133', '00142', '00150', '00154', '00157', '00158', '00165', '00167', '00169', '00172', '00176', '00183', '00184', '00191', '00195', '00201', '00206', '00209', '00211', '00214', '00216', '00221', '00231', '00237', '00238', '00239', '00242', '00247', '00249', '00259', '00266', '00267', '00283', '00289', '00290', '00297', '00300', '00301', '00309', '00310', '00312', '00316', '00320', '00324', '00325', '00327', '00336', '00347', '00351', '00356', '00373', '00382', '00388', '00391', '00392', '00395', '00401', '00417', '00421', '00430', '00433', '00445', '00452', '00455', '00477', '00481', '00495', '00498', '00512', '00563', '00568', '00569', '00587', '00588', '00589', '00591', '00596', '00601', '00605', '00616', '00641', '00654', '00663', '00668', '00682', '00683', '00686', '00687', '00706', '00724', '00727', '00730', '00733', '00742', '00747', '00756', '00759', '00764', '00767', '00774', '00780', '00792', '00797', '00804', '00810', '00814', '00818', '00830', '01004', '01009']
    f1_list = ['00000', '00002', '00005', '00006', '00008', '00011', '00012', '00014', '00020', '00025', '00026', '00028', '00031', '00033', '00035', '00043', '00046', '00052', '00054', '00056', '00058', '00059', '00060', '00062', '00063', '00066', '00068', '00070', '00071', '00074', '00077', '00085', '00087', '00094', '00096', '00098', '00100', '00105', '00106', '00109', '00117', '00128', '00136', '00138', '00139', '00140', '00146', '00155', '00156', '00159', '00160', '00166', '00177', '00178', '00185', '00186', '00188', '00196', '00197', '00203', '00204', '00210', '00212', '00220', '00222', '00230', '00233', '00234', '00235', '00246', '00250', '00253', '00254', '00260', '00263', '00270', '00271', '00273', '00281', '00282', '00284', '00285', '00291', '00293', '00294', '00296', '00303', '00304', '00305', '00306', '00311', '00313', '00317', '00321', '00322', '00328', '00329', '00331', '00332', '00334', '00338', '00340', '00344', '00350', '00352', '00359', '00360', '00364', '00366', '00367', '00369', '00370', '00371', '00383', '00386', '00400', '00403', '00404', '00406', '00409', '00413', '00425', '00426', '00429', '00431', '00436', '00440', '00442', '00443', '00449', '00451', '00456', '00468', '00470', '00472', '00478', '00479', '00480', '00483', '00485', '00488', '00491', '00493', '00494', '00499', '00500', '00501', '00504', '00505', '00506', '00511', '00513', '00516', '00517', '00520', '00523', '00525', '00526', '00528', '00529', '00532', '00537', '00539', '00542', '00543', '00544', '00548', '00550', '00551', '00554', '00556', '00557', '00558', '00559', '00561', '00564', '00570', '00576', '00579', '00582', '00583', '00584', '00586', '00590', '00593', '00594', '00597', '00598', '00599', '00602', '00604', '00606', '00607', '00608', '00612', '00613', '00615', '00618', '00621', '00622', '00625', '00626', '00628', '00631', '00639', '00640', '00650', '00652', '00655', '00656', '00659', '00661', '00674', '00675', '00676', '00677', '00679', '00690', '00691', '00693', '00697', '00698', '00704', '00705', '00708', '00715', '00716', '00718', '00725', '00731', '00732', '00736', '00737', '00739', '00740', '00746', '00750', '00757', '00758', '00760', '00765', '00768', '00772', '00773', '00775', '00777', '00781', '00782', '00784', '00787', '00789', '00791', '00793', '00794', '00795', '00801', '00807', '00811', '00816', '00819', '00823', '00828', '00838', '00840', '00998', '00999', '01000', '01001', '01002', '01003', '01005', '01007', '01008']
    
    for i in range(len(train)):
        if str(train.loc[i, "BraTS21ID"]).zfill(5) not in f0_list and str(train.loc[i, "BraTS21ID"]).zfill(5) not in f1_list:
            train.loc[i, "split"] = 3
            
    index = train.index
    l = index[train["split"]==3].tolist()
    idxs = index[train["split"]!=3].tolist()
    train.drop(l, axis=0, inplace=True)
    
    j = 0
    for i in idxs:
        if str(train.loc[i, "BraTS21ID"]).zfill(5) in f0_list:
            if j in train_idxs:
                train.loc[i, "split"] = 0
            elif j in val_idxs:
                train.loc[i, "split"] = 1
            elif j in test_idxs:
                train.loc[i, "split"] = 2
            #train.loc[i, "BraTS21ID"] = str(train.loc[i, "BraTS21ID"]).zfill(5)
        elif str(train.loc[i, "BraTS21ID"]).zfill(5) in f1_list:
            if j in train_idxs:
                train.loc[i, "split"] = 0
            elif j in val_idxs:
                train.loc[i, "split"] = 1
            elif j in test_idxs:
                train.loc[i, "split"] = 2
            #train.loc[i, "BraTS21ID"] = str(train.loc[i, "BraTS21ID"]).zfill(5)
        j += 1
    
elif args.out_csv_file == "train_h.csv":
    train_idxs = [112, 78, 132, 68, 93, 85, 48, 135, 13, 92, 95, 73, 119, 15, 116, 40, 62, 128, 138, 3, 52, 63, 113, 6, 139, 12, 86, 104, 109, 127, 11, 94, 110, 41, 101, 1, 97, 130, 42, 4, 114, 17, 38, 5, 53, 134, 89, 0, 34, 28, 55, 75, 35, 23, 74, 31, 102, 57, 120, 65, 32, 129, 14, 106, 19, 29, 49, 126, 99, 82, 64, 140, 79, 69, 118, 80, 115, 20, 136, 72, 77, 25, 37, 81, 131, 46, 133, 39, 58, 88, 70, 87, 36, 21, 9, 103, 67, 117, 47]
    
    val_idxs = [45, 60, 7, 51, 66, 27, 71, 54, 123, 8, 76, 16, 125, 122, 124, 98, 105, 83, 33, 56, 91, 22, 137, 24, 2, 111, 26, 121]
    
    test_idxs = [18, 10, 96, 43, 100, 108, 50, 84, 61, 107, 90, 59, 44, 30]
    
    h0_list = ['00009', '00017', '00024', '00032', '00044', '00045', '00049', '00061', '00081', '00084', '00102', '00104', '00110', '00111', '00112', '00113', '00123', '00149', '00151', '00162', '00170', '00192', '00193', '00194', '00217', '00218', '00219', '00227', '00228', '00236', '00241', '00243', '00251', '00258', '00261', '00262', '00269', '00274', '00275', '00280', '00286', '00288', '00298', '00308', '00314', '00318', '00339', '00341', '00343', '00346', '00348', '00349', '00353', '00376', '00377', '00378', '00379', '00380', '00387', '00389', '00390', '00397', '00399', '00402', '00405', '00407', '00410', '00412', '00414', '00418', '00419', '00423', '00432', '00441', '00444', '00446', '00454', '00459', '00464', '00469', '00496', '00507', '00510', '00514', '00518', '00519', '00530', '00533', '00538', '00540', '00545', '00547', '00555', '00565', '00567', '00571', '00572', '00574', '00575', '00578', '00581', '00619', '00620', '00623', '00624', '00630', '00636', '00642', '00645', '00649', '00651', '00657', '00667', '00684', '00685', '00688', '00703', '00709', '00723', '00728', '00729', '00734', '00735', '00744', '00751', '00753', '00778', '00788', '00796', '00799', '00800', '00802', '00803', '00805', '00806', '00809', '00820', '00824', '00836', '00837', '00839']

    for i in range(len(train)):
        if str(train.loc[i, "BraTS21ID"]).zfill(5) not in h0_list:
            train.loc[i, "split"] = 3
            
    index = train.index
    l = index[train["split"]==3].tolist()
    idxs = index[train["split"]!=3].tolist()
    train.drop(l, axis=0, inplace=True)
    
    j = 0
    for i in idxs:
        if str(train.loc[i, "BraTS21ID"]).zfill(5) in h0_list:
            if j in train_idxs:
                train.loc[i, "split"] = 0
            elif j in val_idxs:
                train.loc[i, "split"] = 1
            elif j in test_idxs:
                train.loc[i, "split"] = 2
            #train.loc[i, "BraTS21ID"] = str(train.loc[i, "BraTS21ID"]).zfill(5)
        j += 1

train.to_csv(f"../../RSNA-BTC-Datasets/{args.out_csv_file}", index=False)