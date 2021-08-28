# Nápady a tak
- k A Method for Counting Moving People in Video Surveillance Videos
 - počítají s tím, že pohybující se příznaky musí patřit lidem
  - hrozí, že výsledky budo ovlivněny projíždějícími auty a jinými pohyby v obraze
  - __buď přidat nějakou validaci, že body opravdu patří člověku, nebo použít jinou formu extrakce davu z obrazu (např použít konvoluční neuronovou síť, která bude detekovat, kde se v obraze nachází, či nenachází dav a tímto segmentovat obraz) samotný počet lidí v obraze potom počítat jiným způsobem__

# Data
### <a href=http://www.cvg.reading.ac.uk/PETS2009/a.html__>PETS 2009 benchmark data</a>


# Zdroje

### <a href=https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/231240>A Method for Counting Moving People in Video Surveillance Videos</a>

- Dva přístupy
  - direct (detection based)
    - lidé jsou v obraze detekování (nějakou formou segmentace a detekce objektů) a následně spočítáni  
    - [1]
      - systém odstraní pozadí a následně se snaží matchovat modely lidí s hranami v popředí (Expectation-Maximization algorithm)
      - limitace na nízký počet lidí v "davu"
    - [2]
     - v obraze jsou detekovány příznaky, které jsou trackovány mezi jednotlivými snímky
     - následně dochází ke seskupování bodů do skupin pomocí "Bayesian framework, under the assumption that pairs of points belonging to a same person have a small variance in their mutual distance (quasi-rigid motion)"
     - funguje i s početnějšími davy
     - problémy, dochází-li k pohybu od/ke kameře
   - [3]
    - použití 3D modelu reprezentovaného elipsoidy (hlava, torso, končetiny)
    - použití metody Monte Carlo s pomocí Markovových řetězců k provedení globální optimalizace
      - je to použito na několik snímků
    - dobré na málo a středně zalidněné záběry
    - výpočetně náročné

 - indirect (map based/measurement based)
   - počítání probíhá na základě měření jiné featury, která nevyžaduje detekování jednotlivých osob ve scéně
   - robustnější, jelikož správná detekce všech osob v obraze je ošemetná
     - detekce je problematická například v davu
   - množství pohybujících se pixelů
   - blob size ???
   - dimenze fraktálu??
   - texture features
   - [8] Albiol
    - nejlepší výsledky na PETS2009
    - využití rohů jako feature pointů - Harrisův detektor rohů
    - odstranění rohů patřících pozadí - nějaké prahování na základě velikosti pohybových vektorů mezi jednotlivými snímky
    - počet lidí je odhadnut na základě počtu rohů v obraze - direct corellation
     -  __smoothed mezi několika snímky__

- navržená metoda
  - podobná Albiolovi
  - snaží se opravit její omezení
    - místo Harrisova detektoru se používají SURF příznaky
      - nezávislé na rotaci a škále objektu
    - snaží se brát v potaz hustotu detekovaných příznaků
      - množství okluzí je zvislé na hustotě lidí v obraze
        - málo lidí -> je nepravděpodobné, že dojde k nějaké okluzi
        - moc lidí -> budou se vzájemně překrývat
      - je ale důležité brát v potaz perspektivu
        - stejný počet příznaků blízko kamery může být mnohem méně lidí, než daleko od kamery
        - dochází k seskupování příznaků do skupin
            - vzdálenost clusteru od kamery je odvozena pomocí vzdálenosti nejnižšího a nejvyššího bodu clusteru
    - počet lidí v obraze nyní nekoreluje přímo k množství příznaků
      - není to lineární funkce
      - místo toho je použit estimátor
        - SVR (support vector regressor), založeno na SVM
        - vstup počet příznaků
        - výstup odhad počtu lidí
    - výslede je podobně jako u Albiola protažen low pass filtrem, takže je výsledná funkce v "smoothed" v čase
  - detekce bodů zájmu
    -  předpokládáme, že i klidní lidé se trochu hýbou
      - zajímají nás pouze body s nenulovým pohybovým vektorem
