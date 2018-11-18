# Dota 2 Hero Classification
Neural Network Multi Layer Perceptron

Dota 2 memiliki 115 heroes yang kalau dilihat dari primary attributesnya terbagi dalam 3 kelompok.
Strength (STR), Agility (AGI), Intelligent (INT).

Tiap hero memiliki stats yang berbeda-beda seperti 
Base STR, Base AGI, Base INT, STR Growth, Min DMG, Max DMG, Movement Speed, dll. 
Data yang kita gunakan kali ini berasal dari
https://dota2.gamepedia.com/Table_of_hero_attributes

Data testing dan training sudah disertakan dalam bentuk ZIP.

Berdasarkan nilai stats yang terdapat pada masing-masing atribut, terdapat kendala mengklasifikasikan 
setiap karakter hero karena terdapat sebagaian besar nilai stats pada hero memiliki nilai yang mirip bahkan nilai stats
untuk kategori hero tertentu lebih rendah dari nilai stats kategori hero itu sendiri.

Contoh:
1. Hero A seharusnya adalah kategori AGI, namun nilai stats STR hero tersebut memiliki kemiripan dengan nilai stats AGI hero tersebut.
2. Hero B seharusnya adalah kategori INT, namun nilai stats INT hero tersebut lebih rendah dari nilai stats STR hero tersebut.
dan contoh lainnya yang serupa.

Mengklasifikasikan hero pada Dota 2 dapat dilakukan dengan menggunakan artificial neural network (ANN)
atau Jaringan Saraf Tiruan (JST).

Pembagian data dirincikan sebagai berikut:
1. Total data 115 (hero).
2. Data Training 99 (hero).
3. Data testing 16 (hero).

Hasil yang diperoleh untuk model neural network klasifikasi hero pada Dota 2 menghasilkan akurasi training sebesar 88% 
dan nilai akurasi testing sebesar 75% atau dari 16 hero yang di-testing, terdapat 4 hero yang mengalami kesalahan klasifikasi.
