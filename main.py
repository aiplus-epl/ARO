scanList = ["17DRP5sb8fy",  # 0
            "1LXtFkjw3qL",  # 1
            "1pXnuDYAj8r",  # 2
            "29hnd4uzFmX",  # 3
            "2azQ1b91cZZ",  # 4
            "2n8kARJN3HM",  # 5
            "2t7WUuJeko7",  # 6
            "5LpN3gDmAk7",  # 7
            "5q7pvUzZiYa",  # 8
            "5ZKStnWn8Zo",  # 9
            "759xd9YjKW5",  # 10
            "7y3sRwLe3Va",  # 11
            "8194nk5LbLH",  # 12
            "82sE5b5pLXE",  # 13
            "8WUmhLawc2A",  # 14
            "aayBHfsNo7d",  # 15
            "ac26ZMwG7aT",  # 16
            "ARNzJeq3xxb",  # 17
            "B6ByNegPMKs",  # 18
            "b8cTxDM8gDG",  # 19
            "cV4RVeZvu5T",  # 20
            "D7G3Y4RVNrH",  # 21
            "D7N2EKCX4Sj",  # 22
            "dhjEzFoUFzH",  # 23
            "E9uDoFAP3SH",  # 24
            "e9zR4mvMWw7",  # 25
            "EDJbREhghzL",  # 26
            "EU6Fwq7SyZv",  # 27
            "fzynW3qQPVF",  # 28
            "GdvgFV5R1Z5",  # 29
            "gTV8FGcVJC9",  # 30
            "gxdoqLR6rwA",  # 31
            "gYvKGZ5eRqb",  # 32
            "gZ6f7yhEvPG",  # 33
            "HxpKQynjfin",  # 34
            "i5noydFURQK",  # 35
            "JeFG25nYj2p",  # 36
            "JF19kD82Mey",  # 37
            "jh4fc5c5qoQ",  # 38
            "JmbYfDe2QKZ",  # 39
            "jtcxE69GiFV",  # 40
            "kEZ7cmS4wCh",  # 41
            "mJXqzFtmKg4",  # 42
            "oLBMNvg9in8",  # 43
            "p5wJjkQkbXX",  # 44
            "pa4otMbVnkk",  # 45
            "pLe4wQe7qrG",  # 46
            "Pm6F8kyY3z2",  # 47
            "pRbA3pwrgk9",  # 48
            "PuKPg4mmafe",  # 49
            "PX4nDJXEHrG",  # 50
            "q9vSo1VnCiC",  # 51
            "qoiz87JEwZ2",  # 52
            "QUCTc6BB5sX",  # 53
            "r1Q1Z4BcV1o",  # 54
            "r47D5H71a5s",  # 55
            "rPc6DW4iMge",  # 56
            "RPmz2sHmrrY",  # 57
            "rqfALeAoiTq",  # 58
            "s8pcmisQ38h",  # 59
            "S9hNv5qa7GM",  # 60
            "sKLMLpTHeUy",  # 61
            "SN83YJsR3w2",  # 62
            "sT4fr6TAbpF",  # 63
            "TbHJrupSAjP",  # 64
            "ULsKaCPVFJR",  # 65
            "uNb9QFRL6hY",  # 66
            "ur6pFq6Qu1A",  # 67
            "UwV83HsGsw3",  # 68
            "Uxmj2M2itWa",  # 69
            "V2XKFyX4ASd",  # 70
            "VFuaQ6m2Qom",  # 71
            "VLzqgDo317F",  # 72
            "Vt2qJdWjCF2",  # 73
            "VVfe2KiqLaN",  # 74
            "Vvot9Ly1tCj",  # 75
            "vyrNrziPKCB",  # 76
            "VzqfbhrpDEA",  # 77
            "wc2JMjhGNzB",  # 78
            "WYY7iVyf5p8",  # 79
            "X7HyMhZNoso",  # 80
            "x8F5xyUWy9e",  # 81
            "XcA2TqTSSAj",  # 82
            "YFuZgdQ5vWj",  # 83
            "YmJkqBEsHnH",  # 84
            "yqstnuAEVhm",  # 85
            "YVUC4YcDtcY",  # 86
            "Z6MFQCViBuw",  # 87
            "ZMojNkEp431",  # 88
            "zsNo4HB9uLZ",  # 89
            ]

import open3d as o3d
import numpy as np
import pandas as pd

scanId = scanList[0]

ply_path = f"D:/aro/scan/{scanId}/house_segmentations/{scanId}.ply"

mesh = o3d.io.read_triangle_mesh(ply_path)

# 메쉬를 위에서 본 2D 뷰 생성
vertices = np.asarray(mesh.vertices)

z_values = vertices[:, 2]

# 4분위수 계산
quartiles = np.percentile(z_values, [25, 50, 75])

# Q1 ~ Q3 사이의 정점 필터링
filtered_indices = np.where((z_values > quartiles[0]) & (z_values < quartiles[2]))[0]
filtered_vertices = vertices[filtered_indices]



