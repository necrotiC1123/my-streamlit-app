# -*- coding: utf-8 -*-
"""
TEMEL ANALİZ OTOMASYON TERMİNALİ
================================
Python (veri + matematik) → Streamlit (arayüz) → Claude API (derin analiz)

KURULUM
    pip install -r requirements.txt
    # veya: pip install streamlit yfinance anthropic pandas plotly

ÇALIŞTIRMA
    streamlit run ileriseviye4_v2.py      # (dosya adın neyse o)

CLAUDE API ANAHTARI (yalnızca derin rapor katmanı için gerekli)
    - Ortam değişkeni:  ANTHROPIC_API_KEY
    - veya kenar çubuğundaki alana yapıştır (sadece oturum belleğinde tutulur,
      diske YAZILMAZ, koda gömme!)

DÜRÜSTLÜK NOTU
    Veri kaynağı Yahoo Finance (yfinance). Bu, SEC dipnot derinliği DEĞİLDİR;
    hızlı ve sistematik bir ilk tarama katmanıdır. Sayısal skor bir sezgiseldir
    (heuristik). Nihai hüküm Claude katmanında sorgulanır ve eleştirilir.
    Bu araç eğitim/araştırma amaçlıdır; yatırım tavsiyesi değildir.
"""

import os
import json
import math
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# 0) SABİTLER
# ---------------------------------------------------------------------------

APP_TITLE = "Temel Analiz Otomasyon Terminali"

# Doğrulanmış güncel API model kimlikleri (uygulama içinden "Hesabımdaki
# modelleri listele" ile her zaman canlı liste çekilebilir — ezber yok).
DEFAULT_MODELS = [
    # K1 DÜZELTMESİ (2026-08 denetimi): önceki B1-1 notu gerçeğin TERSİYDİ.
    # "claude-opus-4-8" ve "claude-sonnet-4-6" GEÇERLİ kimliklerdir;
    # "claude-opus-5" / "claude-sonnet-5" diye model YOKTUR (5 ailesinin
    # kimliği "claude-fable-5"tir). Eski liste index=0'da 404 üretiyordu.
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# CAPM sabitleri — Damodaran/NYU Stern (Ocak–Şubat 2026 değerleri).
# Yılda 1-2 kez güncelle: pages.stern.nyu.edu/~adamodar (implied ERP) ve
# 10 yıllık ABD Hazine getirisi. İskonto = Rf + β × ERP (+ ek risk primleri).
# Wong (Nature Methods 2011) renk körü-güvenli palet. Erkeklerin ~%8'i
# kırmızı-yeşil ayırt edemez → durum ASLA yalnız renkle kodlanmaz; her
# rozette ikon + metin de bulunur ("siyah-beyazda da doğru olsun").
WONG = {"turuncu": "#E69F00", "gok": "#56B4E9", "yesil": "#009E73",
        "sari": "#F0E442", "mavi": "#0072B2", "vermilyon": "#D55E00",
        "mor": "#CC79A7", "siyah": "#000000"}
# ── B11-2 (KRİTİK) ────────────────────────────────────────────────────
# Wong paleti AÇIK zemin için tasarlandı; bu uygulama KOYU tema kullanıyor.
# ÖLÇÜM (WCAG 2.1, arka plan #262730 — Streamlit koyu):
#   #000000 siyah      → 1.42:1  ⚠️ metin de grafik de FAIL (görünmez)
#   #0072B2 Wong mavi  → 2.86:1  ⚠️ 3:1 grafik eşiğini bile geçmiyor
#   #D55E00 vermilyon  → 3.84:1  grafik PASS, metin fail
#   #009E73 Wong yeşil → 4.34:1  grafik PASS, metin sınırda
# "bilinmiyor" rozeti siyahtı: koyu zeminde '?' işareti FİİLEN GÖRÜNMÜYORDU.
# Koyu-tema uyarlaması: siyah → açık gri, Wong mavi → açık gök mavisi.
WONG_KOYU = dict(WONG)
WONG_KOYU["siyah"] = "#BDBDBD"       # 1.42:1 → 9.53:1
WONG_KOYU["mavi"] = WONG["gok"]      # 2.86:1 → 6.43:1
DURUM_ROZET = {                       # (ikon, renk) — ikon ZORUNLU
    "iyi": ("✔", WONG["yesil"]), "orta": ("◐", WONG["turuncu"]),
    "kotu": ("✖", WONG["vermilyon"]), "notr": ("•", WONG["gok"]),
    "bilinmiyor": ("?", WONG_KOYU["siyah"])}


def rozet(durum, metin):
    """Renk + İKON + metin — renk körlüğünde de okunur."""
    ik, rk = DURUM_ROZET.get(durum, DURUM_ROZET["bilinmiyor"])
    return f"<span style='color:{rk};font-weight:600'>{ik} {metin}</span>"


# Alım dilimi fiyattan en fazla bu kadar aşağıda olabilir. Daha aşağısı
# 3-4 aylık ufukta ulaşılamaz olduğu için plan üretmez (test bulgusu:
# 6 hissenin 5'inde sinyal HİÇ oluşmuyordu).
DILIM_MAX_ISKONTO = 0.20

# ── RİSKSİZ ORAN — ARTIK OTOMATİK ÇEKİLİYOR ─────────────────────────
# DİKKAT (kavramsal): Fed GECELİK politika faizini belirler; DCF ise
# 10 YILLIK Hazine getirisini kullanır. İkisi aynı şey DEĞİLDİR ve
# birlikte hareket etmek zorunda değildir — 10Y kararı ÖNCEDEN fiyatlar
# ve Fed artırırken bile düşebilir. Doğru refleks "Fed ne yaptı" değil,
# "10Y kaça geldi".
RISK_FREE_RATE = 0.0462      # yedek: ABD 10Y (29 Tem 2026) — canlı çekim
                             # başarısız olursa bu kullanılır
EQUITY_RISK_PREMIUM = 0.0423  # Damodaran implied ERP (Ocak 2026)
RF_ERP_TARIH = "2026-07"     # yedek sabitlerin derlendiği dönem

# Tutma ufku seçenekleri: ay aralığı, ~işlem günü, ~takvim günü.
# 3-4 ay ≈ 3.5 ay: 73 işlem / 107 takvim günü. Kısa ufukta plan, hedeften
# çok TETİK + TAKVİM + ZAMAN STOPUNA yaslanır (dürüst kenar yönetimi).
UFUKLAR = {
    # 3 ay VARSAYILAN: 15 hisse × 5 yıllık geriye dönük testte gün başına
    # verim en yüksek iki seçenekten biri (%0.088/gün; 4 ay %0.103) ve
    # kullanıcının çalışma tarzıyla örtüşen ufuk. Sınır DOLDUĞUNDA çıkılır.
    "3 ay (varsayılan)": {"ay": (3, 3), "islem_gunu": 63, "takvim_gunu": 91},
    "3-4 ay (kısa)":  {"ay": (3, 4),   "islem_gunu": 73,  "takvim_gunu": 107},
    "6-12 ay":        {"ay": (6, 12),  "islem_gunu": 189, "takvim_gunu": 274},
    "1-3 yıl (uzun)": {"ay": (12, 36), "islem_gunu": 504, "takvim_gunu": 730},
}

GLOSSARY = {
    "fiyat": "Borsadaki güncel hisse fiyatı — piyasanın bugünkü teklifi.",
    "adil": "DCF adil değeri: şirketin gelecekte üreteceği nakitlerin bugüne "
            "indirgenmiş toplamı. 'Bu iş, bir nakit makinesi olarak kaça "
            "eder?' sorusunun cevabı. Varsayımlara duyarlıdır — bu yüzden "
            "TEK SAYI değil, kötümser–iyimser bir ARALIK olarak okunmalıdır.",
    "mos": "(Adil değer − fiyat) / adil değer. Pozitifse iskontolu "
           "alıyorsun; negatifse adil değerin ÜSTÜNDE ödüyorsun.",
    "mcap": "Fiyat × hisse sayısı: piyasanın şirketin tamamına biçtiği etiket.",
    "kalite": "Fiyattan BAĞIMSIZ şirket sağlığı (kâr kalitesi, bilanço, "
              "ROIC, iflas mesafesi). 'İyi şirket mi?' sorusu — 'iyi fiyat "
              "mı?' sorusundan ayrıdır; ikisi asla karıştırılmaz.",
    "skor": "0=dip ucuz, 100=balon. Fiyatın adil değere ve çarpanlara göre "
            "konumu — şirketin değil, FİYATIN notu.",
    "pe": "Fiyat/Kâr: 1 $ yıllık kâr için kaç $ ödüyorsun. Yüksekse ya "
          "büyük büyüme fiyatlanmış ya da hisse pahalı.",
    "pfcf": "Fiyat/Serbest Nakit Akışı: kasaya GERÇEKTEN giren nakde göre "
            "fiyat. Kâr makyajlanabilir; nakit çok daha zor.",
    "evebitda": "Şirket Değeri/FAVÖK: borç dahil toplam etiketin kaba "
                "kazanca oranı — farklı borç yapılarını kıyaslamak için.",
    "peg": "F/K'nın büyüme hızına bölünmüşü. ~1 makul, 2+ pahalılık "
           "sinyali. Kaba bir pusula, hassas terazi değil.",
    "ps": "Fiyat/Satış: 1 $ ciro için ödenen fiyat. Kârsız/erken evre "
          "şirketlerde kıyas aracı.",
    "pb": "Fiyat/Defter değeri: muhasebe özsermayesine göre fiyat; "
          "varlık-ağır işlerde (banka, sanayi) daha anlamlı.",
    "piotroski": "9 maddelik temel sağlık testi (kârlılık, borç, "
                 "verimlilik). 7-9 sağlam, 0-3 çürük.",
    "altman": "İstatistiksel iflas mesafesi. 3+ güvenli, 1.81 altı tehlike "
              "bölgesi. Finansal sektörde anlamsızdır.",
    "accrual": "(Net kâr − işletme nakdi) / varlıklar. Negatif = kârın "
               "nakit karşılığı tam; yüksek pozitif = kâğıt üstü kâr riski.",
    "ndebitda": "Net borcun yıllık FAVÖK'e oranı: borcu kaç yıllık kazançla "
                "öder? 3x üstü ağırlaşır, 4x üstü tehlikelidir.",
    "intcov": "Faaliyet kârının faiz giderine oranı: faiz yükünü kaç kez "
              "karşılıyor? 3x altı gergin, 1.5x altı alarm.",
    "cari": "Dönen varlık / kısa vadeli borç. 1'in altı kısa vadeli "
            "sıkışma sinyali (sektöre göre değişir).",
    "rvcagr": "Son 3 yılın yıllık bileşik gelir büyümesi — işin gerçek "
              "motoru budur.",
    "fcfcagr": "Son 3 yılın FCF bileşik büyümesi. Düşük bazdan geliyorsa "
               "(ağır CapEx yılı sonrası) yanıltıcı yüksek görünebilir.",
    "roe": "Net kâr / özsermaye: ortağın parası yılda ne kazandırıyor?",
    "roic": "Vergi sonrası faaliyet kârı / yatırılan sermaye: işin kendisi "
            "sermayeyi yüzde kaçla çalıştırıyor? %15+ güçlüdür.",
    "pos52": "Fiyatın 52 haftalık dip-zirve aralığındaki yeri: %0 dip, "
             "%100 zirve. Tek başına al/sat sinyali DEĞİLDİR; bağlamdır.",
    "psband": "Bugünkü F/S çarpanının kendi tarihsel bandındaki yeri "
              "(yaklaşık hesap). Zirvedeyse getiriyi çarpan genişlemesi "
              "taşımış demektir — o rüzgâr tersine dönebilir.",
    "buyume_var": "Önümüzdeki yıllarda FCF'in yılda ortalama yüzde kaç "
                  "büyüyeceği varsayımı. Değerlemenin en güçlü kolu — "
                  "şirketin gerçek temposuna dayandırılmalı.",
    "iskonto_var": "Geleceğin nakdini bugüne kırarken kullanılan getiri "
                   "eşiği (risk primi dahil). Risk yüksek → iskonto yüksek "
                   "→ adil değer düşük.",
    "terminal_var": "Projeksiyon bittikten sonra sonsuza dek varsayılan "
                    "olgun büyüme (~ekonominin nominal hızı).",
    "yil_var": "Yüksek büyümenin kaç yıl süreceği varsayımı — rekabet "
               "avantajının tahmini ömrü.",
    "beneish": "8 değişkenli muhasebe manipülasyon testi. −2.22 altı temiz, "
               "−1.78 üstü riskli: kârın 'makyajlı' olma OLASILIĞI yüksek "
               "demektir — kanıt değil, adli inceleme çağrısıdır.",
    "revizyon": "Analistlerin kâr tahminlerini hangi yöne çektiği. Yukarı "
                "revizyon rüzgârı arkadan, aşağı revizyon önden eser — fiyat "
                "çoğu zaman revizyonu takip eder.",
    "insider_g": "Yöneticilerin kendi şirket hisselerinde yaptığı alım-satım "
                 "(SEC Form 4). Alım güçlü sinyaldir — kimse batacak gemiye "
                 "para koymaz. Satış vergi/çeşitlendirme olabilir; ama tek "
                 "yönlü yoğun satış yine de not edilir.",
    "katalizor": "Önümüzdeki dönemde fiyatı oynatabilecek takvimli olaylar: "
                 "kazanç açıklaması, temettü tarihleri. Şirkete özel olaylar "
                 "(sözleşme, dava, regülasyon) Claude web katmanında aranır.",
    "epv": "Kazanç Gücü Değeri (Greenwald): şirket hiç büyümese, sadece bugünkü kazancını sonsuza sürdürse ne eder? DCF iyimser senaryoysa EPV TABAN. Fiyat EPV'nin çok üstündeyse değerin çoğu büyüme umudunda — kırılgan.",
    "montecarlo": "Adil değeri tek sayı yerine 10.000 senaryoluk OLASILIK "
                  "dağılımı olarak hesaplar. Bant darsa değerleme sağlam, "
                  "genişse varsayıma aşırı bağımlı demektir.",
    "short": "Float'taki hisselerin yüzde kaçı açığa satılmış. %10+ dikkat, "
             "%20+ yoğun: hem aleyhte bahis hem de yukarı sıkışma (squeeze) "
             "potansiyeli — iki ucu keskin bıçak.",
    "normkaz": "Orta-döngü (medyan) marjla hesaplanan kazanç: tek yılın "
               "zirve/dip yanılsamasından arındırılmış gerçek kazanç gücü. "
               "SEC serisiyle 10 yıla kadar bakar.",
    "fkband": "Bugünkü F/K'nın kendi tarihindeki yeri + analist tahmin "
              "aralığının genişliği. Bandın üstü çoğu zaman çarpan "
              "genişlemesidir (duygu), temel iyileşmesi değil.",
    "cikis": "Girişin aynası: kademeli kâr alma seviyeleri + tezi bozan "
             "'fiyattan bağımsız' çıkış tetikleri + zaman stopu. En pahalı "
             "hata çoğu zaman satmamaktır — plan, duyguyu devreden çıkarır.",
    "rule40": "SaaS kuralı: gelir büyümesi + FCF marjı ≥ %40 ise sağlıklı "
              "denge. 40'ın çok altı = ne büyüyor ne kâr ediyor.",
    "rotce": "Maddi özkaynak getirisi (bankalar için ROE'nin dürüst hali): "
             "şerefiye ve maddi olmayanlar düşülmüş özkaynağa göre kâr. "
             "P/TBV ≈ F/K × ROTCE kimliğiyle çapraz kontrol edilir.",
    "sermaye": "CEO'nun asıl işi: kazanılan parayı nereye koyuyor? Yeniden "
               "yatırım, satın alma, temettü, geri alım, borç ödeme. Buffett "
               "testi: tutulan her 1$ kâr ≥ 1$ piyasa değeri yaratmalı.",
    "bazoran": "Dış görüş: 'benzer şirketler tarihsel olarak ne başardı?' "
               "Piyasanın istediği büyüme tarihsel olarak çok nadirse, hisse "
               "istisna olduğunu KANITLAMALI — varsayılamaz.",
    "kurumsal": "Büyük fonların sahiplik oranı ve en büyük 5 kurumun "
                "çeyreklik hareketi. 13F verisi ~45 gün gecikmelidir — "
                "momentum değil, trend sinyali olarak oku.",
}

# yfinance satır adları sürümden sürüme oynayabildiği için aday listeleriyle
# savunmacı okuma yapılır.
ROWS = {
    "revenue":       ["Total Revenue", "Operating Revenue"],
    "gross":         ["Gross Profit"],
    "op_income":     ["Operating Income", "Total Operating Income As Reported"],
    "net_income":    ["Net Income", "Net Income Common Stockholders",
                      "Net Income Continuous Operations"],
    "ebitda":        ["EBITDA", "Normalized EBITDA"],
    "ebit":          ["EBIT"],
    "interest_exp":  ["Interest Expense", "Interest Expense Non Operating"],
    "cfo":           ["Operating Cash Flow",
                      "Cash Flow From Continuing Operating Activities"],
    "capex":         ["Capital Expenditure"],
    "fcf":           ["Free Cash Flow"],
    "sbc":           ["Stock Based Compensation"],
    "total_debt":    ["Total Debt"],
    "lt_debt":       ["Long Term Debt",
                      "Long Term Debt And Capital Lease Obligation"],
    "cash":          ["Cash Cash Equivalents And Short Term Investments",
                      "Cash And Cash Equivalents"],
    "equity":        ["Stockholders Equity", "Common Stock Equity",
                      "Total Equity Gross Minority Interest"],
    "curr_assets":   ["Current Assets"],
    "curr_liab":     ["Current Liabilities"],
    "total_assets":  ["Total Assets"],
    "total_liab":    ["Total Liabilities Net Minority Interest"],
    "retained":      ["Retained Earnings"],
    "shares":        ["Ordinary Shares Number", "Share Issued"],
    "receivables":   ["Accounts Receivable", "Receivables"],
    "rnd":           ["Research And Development"],
    "inventory":     ["Inventory"],
    "cogs":          ["Cost Of Revenue", "Reconciled Cost Of Revenue",
                      "Cost Of Goods Sold"],
    "payables":      ["Accounts Payable", "Payables",
                      "Payables And Accrued Expenses"],
    "goodwill":      ["Goodwill"],
    "oth_intan":     ["Other Intangible Assets"],
    "gw_intan":      ["Goodwill And Other Intangible Assets"],
    "div_paid":      ["Cash Dividends Paid", "Common Stock Dividend Paid",
                      "Payment Of Dividends"],
    "buyback":       ["Repurchase Of Capital Stock",
                      "Common Stock Payments"],
    "acq":           ["Purchase Of Business",
                      "Net Business Purchase And Sale"],
    "invested_cap":  ["Invested Capital"],
    "tax":           ["Tax Provision", "Income Tax Expense"],
    # NAKİT VERGİ (Damodaran): NOPAT/ROIC defter vergisi değil ÖDENEN
    # vergiyle hesaplanmalı. Doğrudan satır yoksa defter − ertelenmiş.
    "tax_paid":      ["Income Tax Paid Supplemental Data",
                      "Income Taxes Paid", "Taxes Paid",
                      "Income Tax Paid"],
    "tax_deferred":  ["Deferred Income Tax", "Deferred Tax",
                      "Deferred Income Taxes"],
    "pretax":        ["Pretax Income", "Income Before Tax"],
    "dil_shares":    ["Diluted Average Shares", "Basic Average Shares"],
    "ppe":           ["Net PPE", "Properties Plants And Equipment Net"],
    # B5-1: Greenwald'ın bakım-capex yöntemi BRÜT PP&E/Satış oranını kullanır.
    # Net PP&E birikmiş amortisman kadar küçük olduğu için oran da küçük
    # çıkıyor → "büyüme capex'i" olduğundan KÜÇÜK, "bakım capex'i" olduğundan
    # BÜYÜK hesaplanıyordu (ölçüm: 499 vs 58, %763 fark) ve EPV sistematik
    # olarak DÜŞÜK çıkıyordu.
    "ppe_brut":      ["Gross PPE", "Properties Plants And Equipment Gross",
                      "Gross Property Plant And Equipment"],
    "sga":           ["Selling General And Administration",
                      "Selling General And Administrative"],
    "dep":           ["Depreciation And Amortization",
                      "Depreciation Amortization Depletion", "Depreciation"],
    "working_cap":   ["Working Capital"],
}


# ── B9-1: SEKTÖR ÇARPAN MEDYANLARI (Damodaran/NYU Stern, Ocak 2026) ────
# YILDA BİR GÜNCELLE: pages.stern.nyu.edu/~adamodar
# Bu tablo, çarpanların sektörden bağımsız mutlak eşiklerle puanlanması
# hatasını (B9-1) kapatır. Sektör yoksa GENEL medyan kullanılır.
SEKTOR_CARPAN_MEDYAN = {
    "Financial Services":     {"pe": 12.5, "ev_ebitda": None, "p_fcf": 13.0},
    "Utilities":              {"pe": 17.5, "ev_ebitda": 11.5, "p_fcf": 20.0},
    "Energy":                 {"pe": 13.0, "ev_ebitda": 5.2,  "p_fcf": 11.0},
    "Basic Materials":        {"pe": 16.0, "ev_ebitda": 8.5,  "p_fcf": 15.0},
    "Industrials":            {"pe": 22.0, "ev_ebitda": 21.6, "p_fcf": 22.0},
    "Consumer Defensive":     {"pe": 19.0, "ev_ebitda": 12.0, "p_fcf": 20.0},
    "Consumer Cyclical":      {"pe": 18.0, "ev_ebitda": 11.0, "p_fcf": 18.0},
    "Real Estate":            {"pe": 30.0, "ev_ebitda": 20.0, "p_fcf": 25.0},
    "Communication Services": {"pe": 20.0, "ev_ebitda": 10.5, "p_fcf": 18.0},
    "Healthcare":             {"pe": 30.0, "ev_ebitda": 18.0, "p_fcf": 26.0},
    "Technology":             {"pe": 34.0, "ev_ebitda": 24.5, "p_fcf": 30.0},
}
GENEL_CARPAN_MEDYAN = {"pe": 21.0, "ev_ebitda": 14.0, "p_fcf": 20.0}
SEKTOR_MEDYAN_TARIH = "2026-01"


def bolunme_carpani(splits, yil):
    """B3-7: yıldan bugüne bölünmelerin çarpımı.

    forward_bands tarihsel F/K bandını eps_y = net_kâr_y / hisse_y ile
    kuruyor. Fiyat serisi BÖLÜNME DÜZELTMELİ ama hisse sayısı o yılın
    RAPORLANAN (bölünme öncesi) değeri. ÖLÇÜM: 2022'de 4:1 bölünmede
    2021 F/K'sı gerçek 20.0 iken kod 5.0 hesaplıyor (4 KAT düşük) —
    eski yıllar yapay ucuz görünüyor ve "bugünkü F/K bandın ZİRVESİNDE"
    hükmü buradan çıkıyor.
    """
    c = 1.0
    try:
        for ts, oran in (splits or {}).items():
            o = _num(oran)
            if o and o > 0 and pd.Timestamp(ts).year > int(yil):
                c *= float(o)
    except Exception:
        return 1.0
    return c


def son_bar_durumu(hist):
    """B3-8: son bar kısmi/gecikmeli olabilir mi?

    Stop tetiği "son günlük KAPANIŞ" diyerek hist["Close"].iloc[-1]
    okuyor. Piyasa açıkken o bar HENÜZ KAPANMAMIŞTIR ve Yahoo fiyatı
    ~15 dk gecikmelidir — "kapanış stopun altına indi" hükmü gün içi
    geçici bir sarkmayla verilebilir.
    """
    try:
        if hist is None or not len(hist):
            return None
        son = pd.Timestamp(hist.index[-1])
        if getattr(son, "tzinfo", None) is not None:
            son = son.tz_localize(None)
        kismi = son.normalize() >= pd.Timestamp.now().normalize()
        return {"son_bar": str(son.date()), "kismi_olabilir": bool(kismi),
                "not": ("Son bar BUGÜNE ait: piyasa açıksa bu bar HENÜZ "
                        "KAPANMAMIŞTIR ve fiyat ~15 dk gecikmeli olabilir. "
                        "Stop hükmünü seans kapandıktan sonra doğrula."
                        if kismi else "Son bar tamamlanmış seansa ait.")}
    except Exception:
        return None


def distres_kesintisi(z, is_financial=False, esikler=None, model=None):
    """B4-6: iflas riskini İSKONTOYA prim ekleyerek değil DEĞERE kesinti
    uygulayarak modelle.

    İskontoya +%1.5 eklemek (a) risk-ayarlı bir orana risk EKLEMEKTİR
    (çifte sayım), (b) TÜM yılları aynı oranda budar. Damodaran'ın
    distres yaklaşımı olasılık ağırlıklı senaryodur.
    Olasılık haritası SEZGİSELDİR [TÜRETİLMİŞ].
    """
    # ══ ARAŞTIRMA TURU A1 (denetim Bulgu 1a) ═════════════════════════
    # ESKİ harita iki hata taşıyordu:
    #   1) Eşikler MODELDEN bağımsızdı: Z'' (hizmet) skoru üretim
    #      eşikleriyle (1.81/2.99) kesiliyordu — Z'' ölçeği 1.10/2.60.
    #      Kodun DİĞER beş tüketicisi eşiği modele göre okurken bu çağrı
    #      okumuyordu (sessiz fazla-kesinti, hizmetlide %0-11 puan).
    #   2) Doğrusal %0-25 harita PD'nin konveks yapısını yok sayıyordu.
    # YENİ: Altman'ın kendi iki adımlı yöntemi (Altman 2018, J. Credit
    # Risk 14(4)): skor → bono-eşdeğeri rating (BRE) → kümülatif temerrüt
    # oranı; kesinti = PD × (1 − kurtarma). Kurtarma ~%35 (distres satışı
    # defter değerinin ~%25-40'ı; Damodaran %25 kullanır).
    # Z'' BRE tablosu +3.25 SABİTLİ (EM) Z'' ile tanımlıdır; bu dosya
    # sabiti eklemeden hesapladığı için arama öncesi +3.25 kaydırılır.
    # 10Y mortality (1971-2016): BBB %6.3 → %4 · BB %18.2 → %12 ·
    # B %36.7 → %24 · CCC %60 → %39, TAVAN %25 (yalnız B/CCC bölgesinde
    # bağlayıcı — araştırma önerisi). Üretim/Z çapaları [TÜRETİLMİŞ ~]:
    # 1.81 ≈ B bölgesi (Altman 2015: "B ≈ Z 1.6") → tavan; 2.99 ≈ BBB →
    # %4; ~3.6 üstü 0. NOT: bölge ETİKETİ sınıflandırmadır; kesinti PD
    # kalibrasyonludur — "güvenli" bölge PD=0 demek değil, DÜŞÜK demektir.
    zz = _num(z)
    if zz is None or is_financial:
        return 0.0, None
    if model == "uretim":
        p = _ramp(zz, [(1.81, 0.25), (2.99, 0.04), (3.60, 0.0)]) or 0.0
        kaynak = ("Z (üretim) → PD çapaları: 1.81≈B→tavan, 2.99≈BBB→%4 "
                  "[TÜRETİLMİŞ ~; Altman 2018 + Damodaran]")
    elif model in ("hizmet", "hizmet_piyasa_ikamesi"):
        z_adj = zz + 3.25            # BRE tablosu EM-sabitli Z'' ile tanımlı
        p = _ramp(z_adj, [(4.15, 0.25), (4.55, 0.24), (5.40, 0.12),
                          (6.25, 0.04), (6.65, 0.0)]) or 0.0
        kaynak = ("Z''+3.25 → BRE → 10Y mortality × (1−0.35) "
                  "[Altman 2018 Tablo 4/6; Damodaran distres-DCF]")
    else:                            # model bilinmiyorsa eski davranış
        e = esikler or {"tehlike": 1.81, "guvenli": 2.99}
        if zz >= e["guvenli"]:
            return 0.0, None
        p = (0.25 if zz <= e["tehlike"] else
             0.25 * (1 - (zz - e["tehlike"]) /
                     max(e["guvenli"] - e["tehlike"], 1e-9)))
        kaynak = "eski doğrusal harita (model bilinmiyor) [TÜRETİLMİŞ]"
    p = float(min(max(p, 0.0), 0.25))
    if p < 0.005:
        return 0.0, None
    return p, (f"Altman {'Z' if model == 'uretim' else 'Z″'} {zz:.2f} → "
               f"adil değerden %{p*100:.1f} iflas kesintisi. Yöntem: "
               f"{kaynak}. Kesinti = temerrüt olasılığı × (1 − ~%35 "
               f"kurtarma); iskontoya prim eklemek yerine değere kesinti "
               f"(Damodaran) — risk-ayarlı orana risk eklemek çifte "
               f"sayımdır.")


def skor_kararliligi(m, n=400, seed=42):
    """B9-5/6: skorun belirsizlik bandı + hüküm kararlılığı.

    Sert eşikler (35/55/70/85) sürekli bir skoru kategoriye çevirir ve
    sınırda uçurum yaratır. ÖLÇÜM (±3 puan makul girdi gürültüsü):
    skor 55'te hâkim hüküm yalnız %53 kararlı — neredeyse yazı-tura.
    OECD/JRC El Kitabı Adım 7 bu analizi ZORUNLU sayar.
    """
    taban = m.get("score")
    if taban is None:
        # SESSİZ None yerine SEBEBİNİ söyle. Bu turda "fonksiyon var ama
        # çalışmıyor" hatası defalarca çıktı; sessiz dönüşler teşhisi
        # imkânsızlaştırıyor.
        return {"hata": "m['score'] yok — skor hesaplanmadan çağrıldı"}
    try:
        rng = np.random.default_rng(seed)
        ornek = np.clip(taban + rng.normal(0, 3.0, n)
                        + rng.normal(0, 1.5, n), 3, 97)
        hukumler = [verdict_info(float(s), m.get("quality"))[0]
                    for s in ornek]
        hakim = max(set(hukumler), key=hukumler.count)
        kararli = hukumler.count(hakim) / len(hukumler)
        p5, p95 = np.percentile(ornek, [5, 95])
        return {"skor": round(float(taban), 1),
                "bant_p5": round(float(p5), 1),
                "bant_p95": round(float(p95), 1),
                "hukum_kararliligi": round(kararli, 2),
                "hakim_hukum": hakim,
                "not": (f"Skor {taban:.0f}, makul girdi belirsizliği "
                        f"altında {p5:.0f}–{p95:.0f} bandında oynuyor ve "
                        f"hüküm vakaların %{kararli*100:.0f}'inde aynı "
                        f"kalıyor."
                        + (" ⚠️ SINIRDA: hüküm gürültüye duyarlı, tek "
                           "sayıya değil BANDA bak."
                           if kararli < 0.70 else ""))}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 1) GENEL YARDIMCILAR
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def canli_10y():
    """ABD 10 yıllık Hazine getirisini CANLI çek (^TNX).

    Neden gerekli: risksiz oran koda SABİT gömülüydü ve Ocak 2026
    değerindeydi (%4.18). 29 Temmuz 2026'da 10Y %4.62 — 44 baz puanlık
    kayma. ÖLÇÜM: bu fark adil değerleri beta'ya göre %5-9 YÜKSEK
    gösteriyor, yani pahalı bir hisseyi "ucuz" gösterecek büyüklükte.

    ^TNX bazı kaynaklarda yüzde (4.62), bazılarında 10 katı (46.2)
    gelir — makullük süzgeci ikisini de karşılar.
    Başarısız olursa None döner ve modül sabiti devreye girer.
    """
    try:
        import yfinance as yf
        h = _yf_retry(lambda: yf.Ticker("^TNX",
                                        session=_yf_session()
                                        ).history(period="5d"),
                      tries=2, base_wait=1.0)
        if h is None or getattr(h, "empty", True) or "Close" not in h:
            return None
        v = float(h["Close"].dropna().iloc[-1])
        if 12.0 <= v <= 80.0:        # 10 katı biçim (ör. 43.1 → %4.31)
            v /= 10.0
        if not (0.5 <= v <= 12.0):   # makul bant dışı → güvenme
            return None
        try:
            ts = pd.Timestamp(h.index[-1])
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_localize(None)
            tarih = str(ts.date())
        except Exception:
            tarih = None
        return {"oran": v / 100.0, "yuzde": round(v, 3), "tarih": tarih}
    except Exception:
        return None


def rf_erp_guncel():
    """Rf/ERP için TEK KAYNAK — öncelik sırası açık.

    1) Kenar çubuğu override'ı (kullanıcı elle girdiyse o kazanır)
    2) Canlı 10Y (^TNX)
    3) Modül sabiti (yedek)

    B1-8 bulgusu: bu fonksiyon yamada vardı ama ANA DOSYAYA HİÇ
    UYGULANMAMIŞTI; kod sabiti doğrudan okuyordu ve kenar çubuğu
    override'ı yalnız suggest_assumptions'a ulaşıyordu.
    Dönen: (rf, erp, kaynak_metni)
    """
    rf_in = erp_in = None
    try:
        rf_in = _num(st.session_state.get("rf_in"))
        erp_in = _num(st.session_state.get("erp_in"))
    except Exception:
        pass
    erp = (erp_in / 100.0) if erp_in else EQUITY_RISK_PREMIUM
    if rf_in:
        return rf_in / 100.0, erp, "kenar çubuğu (elle girildi)"
    canli = canli_10y()
    if canli:
        return canli["oran"], erp, f"canlı ^TNX ({canli['tarih']})"
    return RISK_FREE_RATE, erp, f"modül sabiti ({RF_ERP_TARIH} — BAYAT)"


def rf_erp_bayatlik():
    """Yedek sabitin üzerinden kaç ay geçti?"""
    try:
        y, a = (int(x) for x in RF_ERP_TARIH.split("-"))
        b = datetime.now()
        return (b.year - y) * 12 + (b.month - a)
    except Exception:
        return None


def _num(x):
    """Her şeyi güvenle float'a çevir; olmuyorsa None."""
    try:
        if x is None:
            return None
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _medyan(dizi):
    """
    GERÇEK medyan (çift uzunlukta iki ortanın ORTALAMASI).
    DENETİM BULGUSU U: kod boyunca `sorted(v)[len(v)//2]` deseni
    kullanılıyordu — bu ÜST ORTAyı alır, medyanı değil. Çift sayıda
    eleman olduğunda sistematik olarak YUKARI sapar: 2 elemanda +%33,
    4 elemanda +%20, 6 elemanda +%17. Etkilediği yerler:
      • normalized_earnings → medyan marj şişer → normalize EPS şişer →
        normalize F/K DÜŞÜK çıkar → çevrimsel zirve tuzağı GİZLENİR
      • forward_bands → tarihsel F/K medyanı yukarı kayar
      • build_peer_table / rakip_ozet → rakip medyanı yukarı kayar
      • dilution_yoy → dilüsyon medyanı yukarı kayar
    Hepsi tek bir doğru fonksiyona bağlandı.
    """
    # B1-9: `dizi or []` numpy dizisinde ValueError atar (MA200 hatasıyla
    # aynı sınıf); ayrıca NaN sıralamayı bozar. İkisi de kapatıldı.
    kaynak = [] if dizi is None else list(dizi)
    v = sorted(x for x in (_num(y) for y in kaynak) if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def fmt_money(x, cur="$"):
    x = _num(x)
    if x is None:
        return "—"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if ax >= div:
            return f"{sign}{cur}{ax / div:,.2f}{unit}"
    return f"{sign}{cur}{ax:,.2f}"


def fmt_pct(x, nd=1):
    x = _num(x)
    return "—" if x is None else f"{x * 100:,.{nd}f}%"


def fmt_x(x, nd=1):
    x = _num(x)
    return "—" if x is None else f"{x:,.{nd}f}x"


def fmt_f(x, nd=2):
    x = _num(x)
    return "—" if x is None else f"{x:,.{nd}f}"


def get_row(df, names):
    """Bir finansal tablodan (satır=etiket, sütun=dönem) aday adlarla satır çek."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    # ── B1-2 (KRİTİK) ─────────────────────────────────────────────────
    # ESKİ "içerir" turu, sözlükteki İLK eşleşmeyi alıyordu → sonuç satır
    # SIRASINA bağlıydı. Kanıt: aynı şirkette faiz gideri 900 vs 120
    # (7.5 KAT) — "Net Interest Expense" brüt sanılıp int_cov'a giriyordu.
    # YENİ: 3 kademe (birebir → ÖNEK → içerir); son iki kademede EN KISA
    # etiket kazanır, eşitlikte alfabetik. Sıradan bağımsız.
    idx = {}
    for i in df.index:
        idx.setdefault(str(i).strip().lower(), []).append(i)

    def _al(orig):
        s = df.loc[orig]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        return pd.to_numeric(s, errors="coerce")

    for n in names:                                    # 1) BİREBİR
        key = n.strip().lower()
        if key in idx:
            return _al(idx[key][0])
    for kosul in (lambda low, k: low.startswith(k),    # 2) ÖNEK
                  lambda low, k: k in low):            # 3) İÇERİR
        for n in names:
            key = n.strip().lower()
            adaylar = [o for low, ol in idx.items() if kosul(low, key)
                       for o in ol]
            if adaylar:
                adaylar.sort(key=lambda o: (len(str(o)), str(o)))
                return _al(adaylar[0])
    return None


def by_year(s):
    """Seri → {yıl:int → değer:float}. Sütunlar Timestamp ya da str olabilir."""
    # B1-6: yfinance sütunları yeniden eskiye sıralıdır; eski kod aynı
    # takvim yılında İLK gördüğünü (yani daha ESKİ dönemi) tutuyordu.
    # Mali yıl değişimi olan şirkette bu, eski dönemi güncel sanmaktır.
    out, damga = {}, {}
    if s is None:
        return out
    for c, v in s.items():
        v = _num(v)
        if v is None:
            continue
        try:
            y = int(getattr(c, "year", None) or str(c)[:4])
        except (TypeError, ValueError):
            continue
        try:
            t = pd.Timestamp(c)
        except Exception:
            t = None
        if y in out:
            onceki = damga.get(y)
            if t is None or onceki is None or t <= onceki:
                continue
        out[y] = v
        damga[y] = t
    return out


def latest(d, k=0):
    """{yıl→değer} sözlüğünden en yeni (k=0) / bir önceki (k=1) değeri al."""
    if not d:
        return None
    ys = sorted(d.keys(), reverse=True)
    return d[ys[k]] if k < len(ys) else None


def cagr(d, n=3):
    """Son n TAKVİM yılı bileşik büyüme; taban/uç pozitif değilse None.
    Üs, liste konumu değil GERÇEK yıl farkıdır — yfinance'te ara yıl eksik
    gelirse büyümeyi şişirme tuzağı böylece kapalıdır."""
    if not d:
        return None
    ys = sorted(d.keys())
    if len(ys) < 2:
        return None
    son = ys[-1]
    within = [y for y in ys[:-1] if son - y <= n]
    eski = min(within) if within else ys[-2]
    span = son - eski
    if span < 1:
        return None
    new, old = d[son], d[eski]
    if new is None or old is None or new <= 0 or old <= 0:
        return None
    return (new / old) ** (1 / span) - 1


def _clean(obj):
    """JSON'a gitmeden önce NaN/inf temizliği (geçersiz JSON üretmemek için)."""
    # B1-10: eski sürüm dict ANAHTARLARINI, np.bool_, pd.Timestamp ve
    # ndarray'i geçirmiyordu → json.dumps ilerideki bir alanda patlıyordu.
    if isinstance(obj, dict):
        return {(k if isinstance(k, (str, int, float, bool, type(None)))
                 else str(k)): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        obj = float(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# 2) VERİ KATMANI (yfinance)
# ---------------------------------------------------------------------------

# ── SEC EDGAR katmanı ────────────────────────────────────────────────────
# Devletin resmî XBRL verisi (data.sec.gov) — ücretsiz, anahtarsız.
# Kural: User-Agent zorunlu (iletişim içermeli), saniyede ≤10 istek.
# Buradaki e-postayı kendi adresinle değiştirebilirsin.
# Ortam değişkeni EDGAR_UA ayarlıysa o kullanılır (gerçek e-posta önerilir).
EDGAR_UA = __import__("os").environ.get(
    "EDGAR_UA",
    "TemelAnalizStreamlit/3.0 (bireysel arastirma; "
    "iletisim: kullanici@example.com)")

# SEC XBRL etiketi adayları: (isim → (namespace, etiketler, birim, tur))
# tur: "sure" = dönemsel akış (yıl boyu), "an" = bilanço anlık değeri
EDGAR_TAGS = {
    "gelir": ("us-gaap",
              ["RevenueFromContractWithCustomerExcludingAssessedTax",
               "Revenues", "SalesRevenueNet",
               "RevenueFromContractWithCustomerIncludingAssessedTax"],
              "USD", "sure"),
    "net_kar": ("us-gaap", ["NetIncomeLoss"], "USD", "sure"),
    # TUR1/TUR3 bulgusu: eski 2. aday (NCI-DAHİL TOPLAM ÖZSERMAYE) azınlık
    # payı DEĞİLDİR — MinorityInterest yoksa EV köprüsüne koca özsermayeyi
    # ekletebiliyordu. Tek doğru etiket bırakıldı.
    "azinlik_payi": ("us-gaap",
                     ["MinorityInterest"], "USD", "an"),
    "imtiyazli": ("us-gaap",
                  ["PreferredStockValue",
                   "PreferredStockLiquidationPreferenceValue"],
                  "USD", "an"),
    "faaliyet_kari": ("us-gaap", ["OperatingIncomeLoss"], "USD", "sure"),
    "cfo": ("us-gaap",
            ["NetCashProvidedByUsedInOperatingActivities"], "USD", "sure"),
    # TUR3: NCI-dahil aday çıkarıldı (ana-hissedar özsermayesini şişirir);
    # etiket yoksa Yahoo özsermayesi zaten yedek olarak devrede kalır.
    "ozsermaye": ("us-gaap",
                  ["StockholdersEquity"], "USD", "an"),
    "seyreltilmis_hisse": ("us-gaap",
                           ["WeightedAverageNumberOfDilutedShares"
                            "Outstanding"], "shares", "sure"),
    "capex": ("us-gaap",
              ["PaymentsToAcquirePropertyPlantAndEquipment"],
              "USD", "sure"),
}


def _edgar_annual(fact_units, unit_key, kind):
    """10-K/FY satırlarını yıl→değer sözlüğüne indir; çeyrek satırları ele."""
    # B2-2: çeyreklikte `filed` dedup vardı, YILLIKTA YOKTU. Aynı dönem
    # yeniden raporlanınca (restatement) sonuç kayıt SIRASINA bağlıydı —
    # testte 4700 vs 5000. Artık EN SON `filed` kazanır.
    aday = {}
    for r in (fact_units.get(unit_key) or []):
        if r.get("form") not in ("10-K", "10-K/A") or r.get("fp") != "FY":
            continue
        end, val = r.get("end"), r.get("val")
        if end is None or val is None:
            continue
        if kind == "sure" and r.get("start"):
            try:
                if (pd.Timestamp(end)
                        - pd.Timestamp(r["start"])).days < 300:
                    continue
            except Exception:
                pass
        filed = r.get("filed") or ""
        onceki = aday.get(end)
        if onceki is None or filed >= onceki[1]:
            aday[end] = (val, filed)
    out = {}
    for end, (val, _f) in aday.items():
        try:
            out[pd.Timestamp(end).year] = float(val)
        except Exception:
            continue
    return out


def _edgar_ceyrek_seri(fu, unit, kind):
    """
    SEC companyfacts → ÇEYREKLİK seri (araştırma reçetesi).
    Tuzaklar ve çözümleri:
      • YTD-vs-çeyrek: 10-Q'da aynı `end` için hem çeyrek (~91g) hem
        yılbaşından-bugüne (~182/273g) kayıt gelir → yalnız 80-100 günlük
        pencereyi çeyrek sayarız.
      • Q4 asla 10-Q ile dosyalanmaz → Q4 = FY − (Q1+Q2+Q3) türetilir.
      • Restatement: aynı dönem birden çok kez raporlanır → en SON
        `filed` tarihli değer kazanır.
      • `fp`/`fy` alanları raporu VEREN dosyalamaya işaret eder, faktın
        kapsadığı döneme değil → dedupe DÖNEM SONU tarihine göre yapılır.
    """
    # ══ B2-1 (PROGRAMIN EN BÜYÜK SESSİZ HATASI) ══════════════════════
    # 10-Q'larda NAKİT AKIŞI kalemleri YALNIZCA yılbaşından-bugüne (YTD)
    # raporlanır. Eski kod 80-100 günlük pencereyi çeyrek sayıyordu; CFO
    # için o pencere neredeyse hiç oluşmadığı için seri 12 yerine 3 döneme
    # düşüyor, ttm_sec None dönüyor ve "SEC EDGAR çeyreklikleri (birincil)"
    # etiketi nakit akışında ASLA gerçekleşmiyordu. Yani DCF'in tabanı olan
    # TTM FCF hiçbir zaman SEC'ten gelmedi.
    # ÇÖZÜM: YTD MERDİVENİ FARKLAMASI — aynı `start`'ı paylaşan kayıtlar
    # sıralanır, ardışık farklar alınır: Q2=H1−Q1, Q3=9A−H1, Q4=FY−9A.
    # Fark 60-130 gün dışındaysa değer ÜRETİLMEZ. Doğrudan 80-100 günlük
    # kayıtlar önceliklidir → gelir tablosunda tek değer değişmez.
    kayitlar = (fu or {}).get(unit) or []
    if kind == "an":
        stok = {}
        for k in kayitlar:
            try:
                son, val = k.get("end"), k.get("val")
                filed = k.get("filed") or ""
                if son is None or val is None:
                    continue
                onceki = stok.get(son)
                if onceki is None or filed >= onceki[1]:
                    stok[son] = (float(val), filed)
            except Exception:
                continue
        return {d: v for d, (v, _) in sorted(stok.items())} or None

    ham = {}
    for k in kayitlar:
        try:
            bas, son, val = k.get("start"), k.get("end"), k.get("val")
            filed = k.get("filed") or ""
            if not bas or son is None or val is None:
                continue
            gun = (pd.Timestamp(son) - pd.Timestamp(bas)).days
            if gun < 25 or gun > 400:
                continue
            onceki = ham.get((bas, son))
            if onceki is None or filed >= onceki[1]:
                ham[(bas, son)] = (float(val), filed, gun)
        except Exception:
            continue
    if not ham:
        return None

    dogrudan, yillik, gruplar = {}, {}, {}
    for (bas, son), (val, _f, gun) in ham.items():
        gruplar.setdefault(bas, []).append((son, gun, val))
        if 80 <= gun <= 100:
            dogrudan[son] = val
        elif 350 <= gun <= 380:
            yillik[son] = val

    turetilmis = {}
    for bas, lst in gruplar.items():          # YTD MERDİVENİ
        lst.sort(key=lambda x: x[0])
        onc_val, onc_gun = 0.0, 0
        for son, gun, val in lst:
            dilim = gun - onc_gun
            if 60 <= dilim <= 130 and son not in dogrudan:
                turetilmis.setdefault(son, val - onc_val)
            onc_val, onc_gun = val, gun

    out = dict(dogrudan)
    for d, v in turetilmis.items():
        out.setdefault(d, v)

    # B2-4: Q4 penceresi 372 gündü → ÖNCEKİ yılın Q4'ünü de kapsıyor,
    # len(ic)==3 hiç tutmuyor ve Q4 HİÇ üretilmiyordu. 358 güne çekildi
    # ve "==3" yerine son 3 çeyrek alınır.
    for yson, yval in yillik.items():
        if yson in out:
            continue
        try:
            ye = pd.Timestamp(yson)
            ybas = ye - pd.Timedelta(days=358)
            ic = sorted((d, v) for d, v in out.items()
                        if ybas < pd.Timestamp(d)
                        < ye - pd.Timedelta(days=20))
            if len(ic) >= 3:
                ic = ic[-3:]
                out[yson] = yval - sum(v for _, v in ic)
        except Exception:
            continue
    return {d: out[d] for d in sorted(out)} or None


def _edgar_parse_ceyreklik(facts):
    """Akış + stok kalemlerinin çeyreklik serileri (dönem sonu → değer)."""
    out = {}
    for name, (ns, tags, unit, kind) in EDGAR_TAGS.items():
        for tag in tags:
            fu = ((((facts or {}).get(ns) or {}).get(tag) or {})
                  .get("units") or {})
            seri = _edgar_ceyrek_seri(fu, unit, kind)
            if seri and len(seri) >= 2:
                out[name] = seri
                break
    return out


def ttm_sec(ceyreklik, ad, min_ceyrek=4):
    """SEC çeyreklik serisinden TTM (son 4 çeyrek toplamı). Eksikse None —
    3 çeyrekten TTM uydurulmaz."""
    seri = (ceyreklik or {}).get(ad) or {}
    if len(seri) < min_ceyrek:
        return None
    son4 = [seri[d] for d in sorted(seri)[-4:]]
    try:                                   # 4 çeyrek gerçekten bitişik mi?
        tarihler = sorted(seri)[-4:]
        yay = (pd.Timestamp(tarihler[-1]) - pd.Timestamp(tarihler[0])).days
        if not (250 <= yay <= 320):        # ~3 çeyrek aralık beklenir
            return None
    except Exception:
        return None
    return float(sum(son4))


def _edgar_parse_yillik(facts):
    # B2-3: eski kod ilk DOLU etiketi bulunca `break` ediyordu. ASC 606
    # geçişinde şirket eski etiketten yeniye geçer; ikisi FARKLI yılları
    # kapsar. Sonuç: 9 yıllık gelir serisi 7 yıla düşüyordu ve bu doğrudan
    # normalized_earnings'in 10 yıllık derinliğini vuruyordu.
    # ARTIK: tüm adaylar BİRLEŞTİRİLİR; çakışan yılda ÖNCELİKLİ etiket
    # (listede önce gelen) kazanır.
    out = {}
    for name, (ns, tags, unit, kind) in EDGAR_TAGS.items():
        birlesik = {}
        for tag in tags:
            fu = ((((facts or {}).get(ns) or {}).get(tag) or {})
                  .get("units") or {})
            seri = _edgar_annual(fu, unit, kind)
            if not seri:
                continue
            for y, v in seri.items():
                birlesik.setdefault(y, v)
        if birlesik:
            out[name] = dict(sorted(birlesik.items()))
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def edgar_cik_map():
    import requests
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers={"User-Agent": EDGAR_UA}, timeout=10)
    r.raise_for_status()
    return {v["ticker"].upper(): int(v["cik_str"])
            for v in r.json().values()}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_edgar_facts(ticker: str) -> dict:
    """SEC CompanyFacts indir + yıllık 10-K kalemlerine indir. Hata → uygulama
    Yahoo ile devam eder; asla analizi düşürmez."""
    import requests
    try:
        cik = edgar_cik_map().get(ticker.upper())
        if not cik:
            return {"durum": "CIK_YOK"}
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        r = requests.get(url, headers={"User-Agent": EDGAR_UA}, timeout=15)
        r.raise_for_status()
        _f = r.json().get("facts", {}) or {}
        # YABANCI DOSYALAYICI (ADR/20-F) TESPİTİ: ifrs-full taksonomisi var
        # ve us-gaap yok/çok seyrekse şirket FPI'dır → 10-Q ve Form 4 YOK.
        _ug = len((_f.get("us-gaap") or {}))
        _if = len((_f.get("ifrs-full") or {}))
        _yabanci = (_if > 0 and _ug < 20)
        return {"durum": "OK", "cik": cik,
                "yillik": _edgar_parse_yillik(_f),
                "ceyreklik": _edgar_parse_ceyreklik(_f),
                "yabanci_dosyalayici": _yabanci,
                "taksonomi": ("ifrs-full" if _yabanci else "us-gaap")}
    except Exception as e:
        return {"durum": "HATA", "hata": str(e)[:120]}


def _parse_filings(subm_json, cik):
    """submissions JSON → son 10-K, son 10-Q, son 2 8-K (form, tarih, URL)."""
    try:
        rec = (subm_json.get("filings") or {}).get("recent") or {}
        forms = rec.get("form") or []
        dates = rec.get("filingDate") or []
        accs = rec.get("accessionNumber") or []
        docs = rec.get("primaryDocument") or []
        out, seen = [], {"10-K": 0, "10-Q": 0, "8-K": 0}
        for f, d, a_, doc in zip(forms, dates, accs, docs):
            key = f if f in seen else None
            if key is None:
                continue
            lim = 2 if key == "8-K" else 1
            if seen[key] >= lim:
                continue
            seen[key] += 1
            acc = str(a_).replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                   f"{acc}/{doc}")
            out.append({"form": f, "tarih": d, "url": url})
        return out or None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_filings(ticker: str):
    """Son SEC dosyalama linkleri (10-K/10-Q/8-K). Hata → None, akış sürer."""
    import requests
    try:
        cik = edgar_cik_map().get(ticker.upper())
        if not cik:
            return None
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        r = requests.get(url, headers={"User-Agent": EDGAR_UA}, timeout=12)
        r.raise_for_status()
        return _parse_filings(r.json(), cik)
    except Exception:
        return None


# ── 8-K Item materyalite haritası (araştırma: kanıt-dereceli) ───────────
FORM8K_ITEMS = {
    "1.01": ("Önemli sözleşme imzalandı", "orta"),
    "1.02": ("Önemli sözleşme sona erdi", "orta"),
    "1.03": ("İflas / konkordato", "kritik"),
    "1.05": ("Materyal siber güvenlik olayı", "kritik"),
    "2.01": ("Varlık alımı/satışı tamamlandı", "orta"),
    "2.02": ("Kazanç sonuçları açıklandı", "yuksek"),
    "2.03": ("Önemli mali yükümlülük doğdu", "orta"),
    "2.04": ("Yükümlülük hızlandırma/temerrüt tetiği", "yuksek"),
    "2.05": ("Yeniden yapılanma maliyeti", "orta"),
    "2.06": ("Materyal değer düşüklüğü (impairment)", "yuksek"),
    "3.01": ("Listeleme uyumsuzluğu / delisting uyarısı", "kritik"),
    "3.02": ("Kayıtsız hisse satışı (seyreltme)", "orta"),
    "4.01": ("Denetçi (muhasebeci) değişimi", "kritik"),
    "4.02": ("Önceki finansallar GÜVENİLMEZ (restatement)", "kritik"),
    "5.01": ("Şirket kontrolü değişti", "yuksek"),
    "5.02": ("CEO/CFO/direktör ayrılışı veya ataması", "yuksek"),
    "5.03": ("Ana sözleşme/tüzük değişikliği", "dusuk"),
    "7.01": ("Regülasyon FD açıklaması", "dusuk"),
    "8.01": ("Diğer olaylar (serbest)", "orta"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_8k_olaylar(ticker: str, limit: int = 12):
    """
    SEC 8-K OLAY AKIŞI (BİRİNCİL, kamu malı): submissions API'den son
    8-K'lar Item numarasına göre sınıflandırılır. Item'lar 'items' alanında
    hazır gelir → dosya içine girmeden materyalite derecelenir. Kırmızı
    bayrak Item'ları (4.02/4.01/3.01/1.05/1.03) öne çıkar.
    """
    import requests
    try:
        cik = edgar_cik_map().get(ticker.upper())
        if not cik:
            return None
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        r = requests.get(url, headers={"User-Agent": EDGAR_UA}, timeout=12)
        r.raise_for_status()
        rec = (r.json().get("filings") or {}).get("recent") or {}
        forms = rec.get("form") or []
        dates = rec.get("filingDate") or []
        items = rec.get("items") or []
        accs = rec.get("accessionNumber") or []
        docs = rec.get("primaryDocument") or []
        out, agirlik = [], {"kritik": 4, "yuksek": 3, "orta": 2, "dusuk": 1}
        for i in range(len(forms)):
            if forms[i] != "8-K":
                continue
            ham = (items[i] if i < len(items) else "") or ""
            kodlar = [x.strip() for x in ham.split(",") if x.strip()]
            etiketler, en_agir, en_sev = [], 0, "dusuk"
            for k in kodlar:
                ad, sev = FORM8K_ITEMS.get(k, ("Diğer bildirim", "dusuk"))
                etiketler.append({"kod": k, "aciklama": ad, "onem": sev})
                if agirlik.get(sev, 1) > en_agir:
                    en_agir, en_sev = agirlik.get(sev, 1), sev
            acc = str(accs[i]).replace("-", "") if i < len(accs) else ""
            doc = str(docs[i]).split("/")[-1] if i < len(docs) else ""
            link = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
                    f"{doc}" if acc and doc else
                    f"https://www.sec.gov/cgi-bin/browse-edgar?action="
                    f"getcompany&CIK={cik:010d}&type=8-K")
            out.append({"tarih": dates[i] if i < len(dates) else None,
                        "itemler": etiketler or [{"kod": "?",
                                                  "aciklama": "Item bilgisi "
                                                  "yok", "onem": "dusuk"}],
                        "en_yuksek_onem": en_sev, "_w": en_agir,
                        "link": link})
            if len(out) >= limit:
                break
        if not out:
            return {"durum": "BOŞ", "cik": cik}
        kirmizi = [o for o in out if o["en_yuksek_onem"] == "kritik"]
        return {"durum": "OK", "cik": cik, "olaylar": out,
                "kirmizi_bayrak": kirmizi or None,
                "not": ("Kaynak: SEC 8-K (birincil, kamu malı). Kırmızı "
                        "bayrak = Item 4.02/4.01/3.01/1.05/1.03 (finansal "
                        "güvenilmezlik, denetçi değişimi, delisting, siber, "
                        "iflas). Bunlar 3-4 aylık ufukta tezi bozabilir; "
                        "her biri 8-K metninden DOĞRULANMALI.")}
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_haberler(ticker: str, limit: int = 12):
    """
    ŞİRKET HABER AKIŞI: yfinance Ticker.news (ücretsiz). Başlık + kaynak +
    özet + tarih + link. Tekrar/kopya başlıklar benzerlikle kümelenir.
    Kişisel kullanım — tam metin gösterilebilir; yön TAHMİNİ YAPILMAZ.
    """
    try:
        import yfinance as yf  # FIX: sessiz NameError tamiri
        tk = yf.Ticker(ticker, session=_yf_session())
        ham = _yf_retry(lambda: tk.news) or []
    except Exception:
        return None
    out = []
    for it in ham:
        c = it.get("content") if isinstance(it, dict) else None
        if isinstance(c, dict):
            baslik = c.get("title")
            ozet = c.get("summary") or c.get("description")
            saglayici = ((c.get("provider") or {}) or {}).get(
                "displayName")
            pd_ = c.get("pubDate") or c.get("displayTime")
            url = (((c.get("canonicalUrl") or {}) or {}).get("url")
                   or ((c.get("clickThroughUrl") or {}) or {}).get("url"))
        else:
            baslik = it.get("title")
            ozet = it.get("summary")
            saglayici = it.get("publisher")
            ts_ = it.get("providerPublishTime")
            pd_ = (pd.Timestamp(ts_, unit="s") if ts_ else None)
            url = it.get("link")
        if not baslik:
            continue
        try:
            ts = pd.Timestamp(pd_)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            tarih = str(ts.date())
            gun = int((pd.Timestamp.now().normalize()
                       - ts.normalize()).days)
        except Exception:
            tarih, gun = None, None
        out.append({"baslik": baslik.strip(),
                    "ozet": (ozet or "").strip()[:600] or None,
                    "kaynak": saglayici, "tarih": tarih, "gun_once": gun,
                    "link": url})
    if not out:
        return {"durum": "BOŞ"}
    # Tekrar tespiti: normalize başlık ilk 60 karakter
    gorulen, tekil = set(), []
    for h in sorted(out, key=lambda x: (x["gun_once"] is None,
                                        x["gun_once"] or 0)):
        anahtar = "".join(ch for ch in (h["baslik"] or "").lower()
                          if ch.isalnum())[:60]
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        tekil.append(h)
    return {"durum": "OK", "haberler": tekil[:limit],
            "not": ("Kaynak: Yahoo/yfinance haber akışı. Başlıklar tekrar "
                    "için filtrelendi. Duygu/yön ETİKETİ YOK — haber tabanlı "
                    "kısa vadeli işlemde bireysel yatırımcı HIZ "
                    "dezavantajındadır; bu akış 'sürpriz olmamak' içindir, "
                    "sinyal değil.")}


def implied_move(bundle_ticker, price, earnings_dt=None):
    """
    OPSİYONDAN İMA EDİLEN HAREKET (araştırma: en yüksek katma değer):
    Kazançtan sonra biten ilk vadedeki ATM straddle'dan piyasanın fiyatladığı
    beklenen % hareket. Formül: (ATM call + ATM put) ÷ fiyat, ×0.85 zaman-
    değeri düzeltmesi. yfinance opsiyon zinciriyle ÜCRETSİZ.
    DÜRÜSTLÜK: straddle×0.85 ≈ 0.68σ BEKLENEN MUTLAK harekettir (~%50
    kapsama) — 1σ DEĞİL, GARANTİ DEĞİL. Ayrıca kazanç öncesi ima edilen
    volatilite şişkindir (IV crush): ima edilen hareket gerçekleşeni
    ortalama ~%15-20 ABARTIR, yani bunu ÜST SINIR gibi oku. Kazanca 1-3 gün
    kala okunan straddle en şişkin değeri verir; piyasa hareketleri
    sistematik olarak ABARTIR (olayların ~%71'i bandın içinde kalır) ve
    kazanç sonrası IV ~%90 oranında söner (IV crush).
    """
    if not price:
        return None
    try:
        import yfinance as yf  # FIX: sessiz NameError tamiri
        tk = yf.Ticker(bundle_ticker, session=_yf_session())
        vadeler = _yf_retry(lambda: tk.options) or []
    except Exception:
        return None
    if not vadeler:
        return {"durum": "OPSIYON_YOK",
                "not": "Bu hisse için opsiyon zinciri bulunamadı "
                       "(opsiyonsuz/likidite düşük olabilir)."}
    bugun = pd.Timestamp.now().normalize()
    hedef = None
    if earnings_dt:
        try:
            ed = pd.Timestamp(earnings_dt).normalize()
            sonra = [v for v in vadeler
                     if pd.Timestamp(v) >= ed]
            hedef = min(sonra) if sonra else None
        except Exception:
            hedef = None
    if hedef is None:                        # kazanç yoksa ilk yakın vade
        ileri = [v for v in vadeler if pd.Timestamp(v) > bugun]
        hedef = min(ileri) if ileri else vadeler[0]
    try:
        oc = _yf_retry(lambda: tk.option_chain(hedef))
        calls, puts = oc.calls, oc.puts
    except Exception:
        return {"durum": "ZINCIR_HATA",
                "not": "Opsiyon zinciri okunamadı."}

    def _atm(df):
        # B3-5: yfinance'te bid/ask YALNIZCA seans saatlerinde dolar.
        # Akşam/hafta sonu bid=ask=0 gelir; lastPrice de 0 ise eski kod
        # mid=0 üretiyor ve program "beklenen hareket %0.0" diye GEÇERLİ
        # GÖRÜNEN bir sayı basıyordu. Ayrıca tek strike, geniş strike
        # aralığında hata üretir → iki-strike interpolasyon eklendi.
        if df is None or df.empty:
            return None
        d = df.copy()
        d["_uz"] = (d["strike"] - price).abs()
        d = d.sort_values("_uz")

        def _mid(row):
            bid = _num(row.get("bid")) or 0.0
            ask = _num(row.get("ask")) or 0.0
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            son = _num(row.get("lastPrice"))
            return son if (son and son > 0) else None

        r0 = d.iloc[0]
        m0 = _mid(r0)
        if m0 is None:
            return None                    # SIFIR STRADDLE ÜRETME
        k0 = _num(r0.get("strike"))
        iv0 = _num(r0.get("impliedVolatility"))
        if iv0 is not None and not (0.01 <= iv0 <= 4.0):
            iv0 = None                     # saçma IV'yi taşıma
        mid, k_eff = m0, k0
        try:
            ters = d[(d["strike"] - price) * (k0 - price) < 0]
            if len(ters) and k0 is not None:
                r1 = ters.iloc[0]
                m1, k1 = _mid(r1), _num(r1.get("strike"))
                if m1 is not None and k1 is not None and abs(k1 - k0) > 1e-9:
                    w = min(max(abs(price - k0) / abs(k1 - k0), 0.0), 1.0)
                    mid = m0 * (1 - w) + m1 * w
                    k_eff = k0 * (1 - w) + k1 * w
        except Exception:
            pass
        return {"strike": k_eff, "mid": mid, "iv": iv0}

    c_, p_ = _atm(calls), _atm(puts)
    if not c_ or not p_ or c_["mid"] is None or p_["mid"] is None:
        return {"durum": "FIYAT_YOK",
                "not": "ATM opsiyon fiyatı alınamadı (işlem saatleri "
                       "dışında bid/ask boş olabilir)."}
    straddle = c_["mid"] + p_["mid"]
    # ── DENETİM BULGUSU V: ETİKET YANLIŞTI ────────────────────────────
    # Black-Scholes'ta ATM straddle ≈ 0.7979 × S × σ√T. Yani straddle/S,
    # BEKLENEN MUTLAK HAREKETtir — 1σ DEĞİL. 1σ = 1.2533 × straddle/S.
    # Kod 0.85 × straddle hesaplıyor (pratisyen geleneği: olay-dışı zaman
    # değerini ayıklar) = ~0.678σ, ki normal dağılımda ~%50 kapsar.
    # Ama not metni "~1σ (~%68)" diyordu → KAPSAMA ABARTILIYORDU.
    # DÜZELTME: sayı korundu (piyasanın kote ettiği rakam budur), etiket
    # düzeltildi ve GERÇEK 1σ ayrı alan olarak eklendi.
    move_usd = straddle * 0.85
    move_pct = move_usd / price
    sigma1_pct = (straddle / price) * 1.2533        # gerçek 1σ
    iv_ort = None
    if c_["iv"] and p_["iv"]:
        iv_ort = (c_["iv"] + p_["iv"]) / 2
    gun = None
    try:
        gun = int((pd.Timestamp(hedef).normalize() - bugun).days)
    except Exception:
        pass
    return {"durum": "OK", "vade": str(hedef), "vade_gun": gun,
            "atm_strike": c_["strike"], "straddle_usd": round(straddle, 2),
            "beklenen_hareket_usd": round(move_usd, 2),
            "beklenen_hareket_pct": round(move_pct, 4),
            "kapsama_yaklasik": "~%50 (beklenen MUTLAK hareket, 1σ değil)",
            "bir_sigma_pct": round(sigma1_pct, 4),
            "bir_sigma_alt": round(price * (1 - sigma1_pct), 2),
            "bir_sigma_ust": round(price * (1 + sigma1_pct), 2),
            "alt_bant": round(price * (1 - move_pct), 2),
            "ust_bant": round(price * (1 + move_pct), 2),
            "atm_iv": round(iv_ort, 4) if iv_ort else None,
            "kazanca_bagli": bool(earnings_dt),
            "not": ("ATM straddle × 0.85 = piyasanın kote ettiği BEKLENEN "
                    "MUTLAK HAREKET. DİKKAT: bu ~1σ DEĞİLDİR — normal "
                    "dağılımda yaklaşık %50'yi kapsar (Black-Scholes'ta ATM "
                    "straddle ≈ 0.80σ, ×0.85 ile ≈0.68σ). Gerçek 1σ (~%68 "
                    "kapsama) 'bir_sigma_*' alanlarındadır ve DAHA "
                    "GENİŞTİR. Piyasa hareketleri sistematik olarak ABARTIR; "
                    "kazanç sonrası IV ~%90 söner (IV crush). Yön TAHMİNİ "
                    "değildir.")}


def _et_bul(el, ad):
    """Ad-alanı (namespace) bağımsız ilk eşleşen düğümü bul."""
    if el is None:
        return None
    for c in el.iter():
        if c.tag.split("}")[-1] == ad:
            return c
    return None


def _et_txt(el, ad):
    c = _et_bul(el, ad)
    return (c.text or "").strip() if c is not None and c.text else None


def _form4_xml_parse(xml_text):
    """
    SEC Form 4 XML → içeriden işlem listesi (BİRİNCİL KAYNAK).
    İşlem kodları (SEC şeması): P=açık piyasa ALIMI, S=satış, A=ödül/grant,
    M=opsiyon kullanımı, G=hediye, F=vergi stopajı, C=dönüşüm, D=elden çık.
    aff10b5One işaretliyse işlemler ÖNCEDEN PLANLIDIR (Rule 10b5-1) —
    planlı satışın sinyal gücü düşüktür; Yahoo özetinde bu ayrım yoktur.
    """
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    isim, rol = None, []
    ro = _et_bul(root, "reportingOwner")
    if ro is not None:
        isim = _et_txt(ro, "rptOwnerName")
        rel = _et_bul(ro, "reportingOwnerRelationship")
        if rel is not None:
            if (_et_txt(rel, "isDirector") or "0") in ("1", "true"):
                rol.append("Yönetim Kurulu")
            if (_et_txt(rel, "isOfficer") or "0") in ("1", "true"):
                rol.append(_et_txt(rel, "officerTitle") or "Yönetici")
            if (_et_txt(rel, "isTenPercentOwner") or "0") in ("1", "true"):
                rol.append("%10+ ortak")
    planli = any((e.text or "").strip().lower() in ("1", "true")
                 for e in root.iter()
                 if e.tag.split("}")[-1] == "aff10b5One")
    KOD = {"P": "ALIM", "S": "SATIM", "A": "ÖDÜL", "M": "OPSİYON",
           "G": "HEDİYE", "F": "VERGİ", "C": "DÖNÜŞÜM", "D": "ELDEN ÇIK."}

    def _val(tr, ad):
        n = _et_bul(tr, ad)
        return _et_txt(n, "value") if n is not None else None

    isl = []
    for tr in root.iter():
        if tr.tag.split("}")[-1] != "nonDerivativeTransaction":
            continue
        kod = _et_txt(tr, "transactionCode")
        adet = _num(_val(tr, "transactionShares"))
        fiyat = _num(_val(tr, "transactionPricePerShare"))
        isl.append({"tarih": _val(tr, "transactionDate"), "kod": kod,
                    "tur": KOD.get(kod, "DİĞER"),
                    "adet": adet, "fiyat": fiyat,
                    "deger_usd": (adet * fiyat
                                  if adet and fiyat else None)})
    if not isl:
        return None
    return {"kisi": isim, "unvan": ", ".join(rol) or None,
            "planli_10b5_1": planli, "islemler": isl}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_form4_ozet(ticker: str, limit: int = 10):
    """
    İçeriden işlemlerde BİRİNCİL kaynak: SEC submissions'tan son Form 4
    dosyalamaları alınır, XML'leri işlem koduna göre ayrıştırılır. Yahoo
    özeti seyrek/gecikmeli olabildiği ve 10b5-1 ayrımı taşımadığı için
    bu yol esastır; hata olursa None döner ve akış Yahoo yedeğine düşer.
    SEC nezaketi: istekler aralıklı, User-Agent zorunlu.
    """
    import requests, time
    try:
        cik = edgar_cik_map().get(ticker.upper())
        if not cik:
            return None
        link = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&CIK={cik:010d}&type=4&dateb=&owner=include&count=40")
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        r = requests.get(url, headers={"User-Agent": EDGAR_UA}, timeout=12)
        r.raise_for_status()
        rec = (r.json().get("filings") or {}).get("recent") or {}
        satirlar = []
        for f, d, a_, doc in zip(rec.get("form") or [],
                                 rec.get("filingDate") or [],
                                 rec.get("accessionNumber") or [],
                                 rec.get("primaryDocument") or []):
            if f not in ("4", "4/A"):
                continue
            doc = str(doc).split("/")[-1]     # xsl.../x.xml → x.xml (ham)
            if not doc.lower().endswith(".xml"):
                continue
            acc = str(a_).replace("-", "")
            satirlar.append((d, "https://www.sec.gov/Archives/edgar/data/"
                                f"{cik}/{acc}/{doc}"))
            if len(satirlar) >= limit:
                break
        if not satirlar:
            return {"durum": "BOŞ", "cik": cik, "link": link}
        dosyalar = []
        for d, u in satirlar:
            try:
                rx = requests.get(u, headers={"User-Agent": EDGAR_UA},
                                  timeout=12)
                rx.raise_for_status()
                p = _form4_xml_parse(rx.text)
                if p:
                    p["dosya_tarihi"] = d
                    dosyalar.append(p)
            except Exception:
                pass
            time.sleep(0.15)                  # 10 istek/sn sınırının altı
        if not dosyalar:
            return {"durum": "BOŞ", "cik": cik, "link": link}
        return {"durum": "OK", "cik": cik, "link": link,
                "dosyalar": dosyalar}
    except Exception:
        return None


def edgar_crosscheck(m, ed):
    """
    Yahoo ↔ SEC kalem kalem karşılaştırma. Uyum: ✓ ≤%2 (resmî teyit),
    ≈ %2-10 (dönem farkı olabilir), ✗ >%10 (SEC ESASTIR).
    """
    if not ed:
        return {"durum": "KAPALI"}
    if ed.get("durum") != "OK":
        return {"durum": ed.get("durum", "YOK"), "hata": ed.get("hata")}
    yy = ed.get("yillik") or {}
    pairs = []

    def pair(ad, yf_val, name, y):
        d = yy.get(name) or {}
        if yf_val is None or not d:
            return
        sec_y = y if y in d else max(d)
        sv = d.get(sec_y)
        if sv is None:
            return
        dev = abs(yf_val - sv) / max(abs(sv), 1e-9)
        uyum = "✓" if dev <= 0.02 else ("≈" if dev <= 0.10 else "✗")
        pairs.append({"kalem": ad, "yahoo": round(yf_val, 2),
                      "sec": round(sv, 2), "sec_yil": int(sec_y),
                      "sapma_pct": round(dev * 100, 1), "uyum": uyum})

    for ad, seri, name in (("Gelir (yıllık)", m.get("rev"), "gelir"),
                           ("Net kâr (yıllık)", m.get("ni"), "net_kar"),
                           ("CFO (yıllık)", m.get("cfo"), "cfo"),
                           ("Seyreltilmiş hisse (ort.)",
                            m.get("dil_shares_seri"), "seyreltilmis_hisse")):
        s = seri or {}
        if s:
            y0 = max(s)
            pair(ad, s[y0], name, y0)
    rev_s = m.get("rev") or {}
    if m.get("equity") is not None and rev_s:
        pair("Özsermaye", m["equity"], "ozsermaye", max(rev_s))

    teyit = sum(1 for p in pairs if p["uyum"] == "✓")
    return {"durum": "OK", "cik": ed.get("cik"),
            "ozet": f"{teyit}/{len(pairs)} kalem SEC 10-K ile birebir "
                    f"uyumlu (≤%2)",
            "karsilastirma": pairs,
            "kaynak": "SEC EDGAR CompanyFacts (resmî XBRL)"}


_YF_SES = "UNSET"


def _yf_session():
    """curl_cffi tarayıcı-taklidi oturumu (varsa) — Yahoo'nun 2024-2026
    rate-limit sertleşmesine karşı rapor önerisi. Kütüphane kurulu değilse
    None döner ve yfinance normal oturumuyla sürer (sessiz bozulma yok)."""
    global _YF_SES
    if _YF_SES == "UNSET":
        try:
            from curl_cffi import requests as _cffi
            _YF_SES = _cffi.Session(impersonate="chrome")
        except Exception:
            _YF_SES = None
    return _YF_SES


def _yf_retry(fn, tries=3, base_wait=1.5):
    """yfinance çağrısını 429/rate-limit/ağ hatasında üstel bekleme ile
    tekrar dene (Kaynak-2 tuzağı #1: retry/sleep/cache savunması).
    Rate-limit dışı hatalar aynen yükseltilir — sessiz yutma YOK."""
    # B3-2: yfinance artık TİPLİ YFRateLimitError fırlatıyor. Metin
    # eşleşmesi bugün çalışıyor çünkü mesajda tesadüfen "Too Many Requests"
    # geçiyor — kütüphane mesajı yeniden yazarsa eşleşme SESSİZCE ölür ve
    # hız sınırı ÖLÜMCÜL hataya döner. Artık önce TİP, sonra metin.
    import time
    import random
    try:
        from yfinance.exceptions import YFRateLimitError as _YFRate
    except Exception:
        _YFRate = None
    ANAHTAR = ("429", "too many", "rate limit", "rate-limit", "timed out",
               "temporarily", "connection", "expecting value")
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            tipli = bool(_YFRate is not None and isinstance(e, _YFRate))
            if not tipli and not any(k in str(e).lower() for k in ANAHTAR):
                raise
            time.sleep(base_wait * (2 ** i) + random.uniform(0, 0.4))
    raise last


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bundle(ticker: str, use_edgar: bool = True) -> dict:
    """Ticker için tüm ham veriyi tek pakette indir (30 dk önbellek)."""
    import yfinance as yf

    tk = yf.Ticker(ticker, session=_yf_session())

    info = {}
    for getter in ("get_info", "info"):
        try:
            obj = getattr(tk, getter)
            info = _yf_retry(lambda o=obj: (o() if callable(o) else o))
            if info:
                break
        except Exception:
            info = {}

    def fast(*keys):
        for k in keys:
            try:
                v = getattr(tk.fast_info, k)
                if v is not None:
                    return v
            except Exception:
                pass
            try:
                v = tk.fast_info[k]
                if v is not None:
                    return v
            except Exception:
                pass
        return None

    def grab(attr):
        try:
            df = _yf_retry(lambda: getattr(tk, attr))
            return df if isinstance(df, pd.DataFrame) and not df.empty else None
        except Exception:
            return None

    inc, bal, cf = grab("income_stmt"), grab("balance_sheet"), grab("cashflow")
    qcf = grab("quarterly_cashflow")

    # ══ B3-1 (KRİTİK) ════════════════════════════════════════════════
    # auto_adjust=True hem bölünme HEM TEMETTÜ düzeltir. Temettü düzeltmesi
    # geçmiş fiyatların TAMAMINI aşağı kaydırır — ve bu seri destek/direnç,
    # ATR stopu, 52 hafta dip/zirve, MA'lar ve KULLANICIYA GÖSTERİLEN
    # grafik için kullanılıyordu. ÖLÇÜM (%3 verim, 5 yıl): 3 yıl önceki dip
    # nominal 66.25 → düzeltilmiş 60.99 (−%8.6). Program "destek 60.99"
    # derken kullanıcı grafiğinde 66.25 görüyor; emir yanlış yere konuyor.
    # ÇÖZÜM: TEK istekten İKİ baz. auto_adjust=False ile:
    #   Close/High/Low = bölünme düzeltmeli, TEMETTÜ DÜZELTMESİZ → seviyeler
    #   "Adj Close"     = bölünme + temettü → toplam getiri (vol/momentum)
    # Ek istek maliyeti YOK.
    hist, hist_tr, _tem_etki = None, None, None
    try:
        _h = _yf_retry(lambda: tk.history(period="5y", auto_adjust=False))
        if _h is not None and not getattr(_h, "empty", True):
            hist = _h.drop(columns=[c for c in ("Adj Close",) if c in _h])
            if "Adj Close" in _h and "Close" in _h:
                _ac = _h["Adj Close"].dropna()
                _c = _h["Close"].dropna()
                hist_tr = _ac
                _ortak = _ac.index.intersection(_c.index)
                if len(_ortak) > 10:
                    _tem_etki = float(_ac.loc[_ortak[0]]
                                      / _c.loc[_ortak[0]] - 1)
    except Exception:
        hist = None
    _hist_notu = None
    if hist is None:
        # ── STOOQ YEDEĞİ (TUR1 bulgusu): _stooq_fiyat tanımlıydı ama HİÇ
        # çağrılmıyordu — Yahoo çökünce fiyat serisi tamamen boş kalıyordu.
        # Yalnız kapanış gelir: ATR yaklaşık, hacim/4 saatlik yok.
        hist = _stooq_fiyat(ticker)
        if hist is not None and len(hist):
            _hist_notu = ("Fiyat serisi STOOQ yedeğinden geldi (Yahoo "
                          "vermedi): yalnız kapanış — hacim ve 4 saatlik "
                          "yok, ATR kapanıştan YAKLAŞIK.")
        else:
            hist = None

    price = (_num(info.get("currentPrice"))
             or _num(info.get("regularMarketPrice"))
             or _num(fast("last_price", "lastPrice")))
    if price is None and hist is not None:
        price = _num(hist["Close"].dropna().iloc[-1])

    if price is None and inc is None and bal is None:
        raise ValueError(
            f"'{ticker}' için hiçbir veri alınamadı. Ticker'ı kontrol edin "
            f"(ABD hisseleri: AAPL, NVDA...). Yahoo bazen oran sınırlar; "
            f"1-2 dk sonra tekrar deneyin veya `pip install -U yfinance` çalıştırın."
        )

    # Grup 2 katmanları için ek veri — hepsi savunmacı, biri patlarsa None
    extras = {}
    for name, fn in (("calendar", lambda: tk.calendar),
                     ("earnings_dates", lambda: tk.get_earnings_dates(limit=12)),
                     ("insider_tx", lambda: tk.insider_transactions),
                     ("eps_trend", lambda: tk.eps_trend),
                     ("eps_rev", lambda: tk.eps_revisions),
                     ("price_targets", lambda: tk.analyst_price_targets),
                     ("recs", lambda: tk.recommendations),
                     ("earn_est", lambda: tk.earnings_estimate),
                     ("rev_est", lambda: tk.revenue_estimate),
                     ("growth_est", lambda: tk.growth_estimates),
                     ("major_holders", lambda: tk.major_holders),
                     ("inst_holders", lambda: tk.institutional_holders)):
        try:
            extras[name] = fn()
        except Exception:
            extras[name] = None

    extras["edgar"] = (fetch_edgar_facts(ticker) if use_edgar
                       else {"durum": "KAPALI"})
    try:
        extras["filings"] = fetch_filings(ticker) if use_edgar else None
    except Exception:
        extras["filings"] = None
    try:
        extras["form4"] = fetch_form4_ozet(ticker) if use_edgar else None
    except Exception:
        extras["form4"] = None
    try:
        extras["olaylar_8k"] = fetch_8k_olaylar(ticker) if use_edgar else None
    except Exception:
        extras["olaylar_8k"] = None
    try:
        extras["haberler"] = fetch_haberler(ticker)
    except Exception:
        extras["haberler"] = None

    try:
        _sp = tk.splits
        _bolunmeler = ({str(pd.Timestamp(k).date()): float(v)
                        for k, v in _sp.items()}
                       if _sp is not None and len(_sp) else {})
    except Exception:
        _bolunmeler = {}

    return {
        "ticker": ticker,
        "info": dict(info) if info else {},
        "fast": {"price": price,
                 "mcap": _num(fast("market_cap", "marketCap")),
                 "shares": _num(fast("shares"))},
        **extras,
        "inc": inc, "bal": bal, "cf": cf, "qcf": qcf,
        # ── B3-7 ZİNCİRİ: bölünme verisi HİÇ ÇEKİLMİYORDU ────────────
        # bolunme_carpani() yazılmıştı ama girdi kaynağı yoktu → ölü kod.
        # Fiyat serisi bölünme-düzeltmeli, SEC hisse adedi DEĞİL; 4:1
        # bölünmede eski yılların F/K'sı 4 KAT yanlış çıkıyordu.
        "splits": _bolunmeler,
        "hist": hist, "hist_kaynak_notu": _hist_notu,
        # B3-1: iki fiyat bazı — seviyeler vs toplam getiri
        "hist_tr": hist_tr, "temettu_etkisi": _tem_etki,
        "fiyat_bazi": ("bölünme düzeltmeli, TEMETTÜ DÜZELTMESİZ — broker "
                       "grafiğiyle eşleşir"),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def ttm_sum(qdf, names, min_q=4):
    """Çeyreklik toplam (TTM). O4: 4 çeyreğin ~12 aya yayıldığı da denetlenir
    (ttm_sec paritesi); ortadaki çeyrek eksikse 15+ aylık sahte-TTM,
    DCF tabanına sızmasın."""
    s = get_row(qdf, names)
    if s is None:
        return None
    cift = [(i, v) for i, v in ((i, _num(x)) for i, x in s.items())
            if v is not None][:4]
    if len(cift) < min_q:
        return None
    try:
        ds = [pd.Timestamp(i) for i, _ in cift]
        if not (250 <= (max(ds) - min(ds)).days <= 320):
            return None
    except Exception:
        pass  # etiket tarih değilse eski davranış
    return sum(v for _, v in cift)


# ---------------------------------------------------------------------------
# 3) DEĞERLEME MOTORU — DCF, TERS-DCF, PEG
# ---------------------------------------------------------------------------

def oto_buyume(m, taban=0.08):
    """
    ŞİRKETİN KENDİ GEÇMİŞİNDEN büyüme varsayımı.
    Neden: sabit %8 varsayımı, nakit akışı hızlı büyüyen kaliteli
    şirketlerde adil değeri sistematik olarak DÜŞÜK gösteriyordu — 5
    hisselik testte program hiçbirine "al" diyemedi (MSFT'te izleme eşiği
    fiyatın %158 altındaydı).
    Yöntem: FCF ve gelir CAGR'ının DÜŞÜĞÜ alınır (temkinli); TABAN −%5
    (küçülen şirket küçülen kabul edilir), TAVAN kaliteye göre %15/18/22.
    Veri yetersizse %8 jenerik taban döner. (Aşağıdaki düzeltme notu
    esastır; bu özet ona göre güncellendi.)
    """
    aday = []
    for anahtar in ("fcf", "rev"):
        seri = m.get(anahtar)
        if not isinstance(seri, dict) or len(seri) < 3:
            continue
        yillar = sorted(seri)
        ilk, son = _num(seri[yillar[0]]), _num(seri[yillar[-1]])
        n = yillar[-1] - yillar[0]
        if not ilk or not son or ilk <= 0 or son <= 0 or n < 2:
            continue
        try:
            aday.append((son / ilk) ** (1.0 / n) - 1.0)
        except Exception:
            continue
    if not aday:
        return taban, "veri yetersiz — jenerik %8 kullanıldı"
    # ── B4-3 ─────────────────────────────────────────────────────────
    # MIN almak "temkin" DEĞİL, SİSTEMATİK AŞAĞI YANLILIKTIR (iki gürültülü
    # tahminin minimumunun beklenen değeri her ikisinin de altındadır).
    # ÖLÇÜM (20.000 simülasyon, gerçek %10): MIN → %7.68 (2.32 puan aşağı),
    # HARMAN → %10.00 (yansız); değer etkisi %+8.1. Daha kötüsü: marjı
    # GENİŞLEYEN kaliteli şirkette FCF-CAGR > gelir-CAGR olur ve MIN tam da
    # o kalite sinyalini çöpe atar — "kaliteli şirket hep pahalı çıkıyor"
    # şikâyetinin ana bileşenlerinden biri buydu.
    g = (sum(aday) / len(aday)) if len(aday) > 1 else aday[0]
    # TAVAN KALİTEYE GÖRE (denetim düzeltmesi): sabit %15 tavan, yüksek
    # ROIC'li şirketleri sistematik olarak kesiyordu. Yüksek getirili
    # şirket büyümesini daha uzun sürdürebilir → tavan %22'ye çıkar.
    # B4-4: sürdürülebilirlik kimliği g = ROIC × yeniden-yatırım oranı.
    # Eski tavanlar İMKÂNSIZDI: ROIC %20 → tavan %22 ⇒ %110 yeniden
    # yatırım; ROIC %12 → %18 ⇒ %150. Artık tavan = ROIC × 0.8.
    _roic = _num(m.get("roic"))
    _tavan = max(0.08, min(0.25, (_roic if _roic and _roic > 0 else 0.12)
                           * 0.8))
    # ── DENETİM BULGUSU T: TABAN İYİMSERDİ ────────────────────────────
    # Eski kod `max(0.0, ...)` ile negatif büyümeyi SIFIRA çekiyordu.
    # Docstring "temkinli" diyor ama bu tam tersi: küçülen bir şirkete
    # "büyümeyecek ama küçülmeyecek de" varsayımı YAPMAK, onu olduğundan
    # DEĞERLİ gösterir (testte −%9.1 CAGR → %0 atanıyordu).
    # DÜZELTME: taban −%5 (kardeş fonksiyon suggest_assumptions ile aynı).
    # DCF'in fade mantığı zaten negatiften terminale doğru YUKARI çıkar —
    # yani "küçülüp sonra istikrara kavuşma" senaryosu doğru modellenir.
    g_k = max(-0.05, min(g, _tavan))
    not_ = (f"şirketin kendi geçmişinden (%{g*100:.1f} ham"
            + (f", %{g_k*100:.1f}'e kıstırıldı)" if abs(g_k - g) > 1e-9
               else ")"))
    if g < 0:
        not_ += (" — ŞİRKET KÜÇÜLÜYOR: negatif büyüme DCF'e olduğu gibi "
                 "giriyor (−%5 tabanla); 'küçülme durur' varsayımı "
                 "yapılmadı.")
    return g_k, not_


def terminal_roic_sec(roic, discount):
    """ARAŞTIRMA TURU A2 (denetim Bulgu 3): terminal RONIC seçimi.

    B4-2 kısıtı yazılmış ve ölçülmüştü ama roic_terminal HİÇBİR çağrıda
    geçilmediği için ölü koddu — bu fonksiyonla BAĞLANIYOR.
    Varsayılan RONIC = iskonto (rekabetçi denge: yeni yatırım değer
    yaratmaz; McKinsey continuing-value bölümü, Damodaran: fazla getiri
    yoksa terminal büyüme değere ETKİSİZDİR). Mevcut ROIC iskontonun
    üstündeyse (moat vekili) primin YARISI tanınır, tavan +5 puan —
    McKinsey geçiş matrisleri: en üst ROIC beşliği bile 10 yılda ~%12-13'e
    sönümleniyor; kalıcı prim tarihsel olarak ~4-6 puanla sınırlı.
    [Koller/Goedhart/Wessels; Damodaran terminal value; Mauboussin
    Base Rate Book — 0.5 katsayısı muhafazakâr mühendislik seçimi.]
    """
    d = _num(discount)
    if not d or d <= 0:
        return None
    r = _num(roic)
    if r is None or r <= d:
        return d
    return min(r, d + min(0.05, 0.5 * (r - d)))


_ORTA_YIL = True        # B4-1: orta-yıl konvansiyonu (kapatmak için False)


def dcf_fair_value(fcf0, growth, discount, terminal, years,
                   net_debt, shares, adjust_net_debt=True,
                   roic_terminal=None):
    """
    Basit iki aşamalı FCF-DCF.
    Not: FCF = CFO − CapEx pratikte kaldıraçlı (FCFE-benzeri) bir akıştır; net
    borcu ayrıca düşmek bilinçli olarak TEMKİNLİ bir sadeleştirmedir (borçlu
    şirkete çifte ceza, nakit zengine hak teslimi). Kenar çubuğundan kapatılabilir.
    """
    fcf0, shares = _num(fcf0), _num(shares)
    if not fcf0 or fcf0 <= 0 or not shares or shares <= 0:
        return None
    if discount <= terminal:
        return None
    # ── KADEMELİ GEÇİŞ (denetim düzeltmesi) ───────────────────────────
    # ESKİ: `years` yıl sabit `growth`, sonra ani terminal düşüş. Bu, geniş
    # hendekli şirketlerin üstün getirisini terminalde ANİDEN sıfırlıyor ve
    # onları sistematik olarak DÜŞÜK değerliyordu (Damodaran: terminal
    # dönemde ROIC'i WACC'a çökertmek kaliteli şirketi ucuz gösterir).
    # YENİ: ilk yarı yüksek büyüme, ikinci yarı DOĞRUSAL olarak terminale
    # iner (fade). Böylece geçiş ani değil kademeli olur.
    # ── B4-1: ORTA-YIL KONVANSİYONU ───────────────────────────────────
    # Eski sürüm nakdi YIL SONUNDA varsayıyordu. Nakit yıl boyunca dağılır;
    # standart pratik (McKinsey, Damodaran, ABD adillik görüşlerinin büyük
    # çoğunluğu) orta-yıldır: (1+d)^(t−0.5). ÖLÇÜM: %10 iskontoda +%4.88 —
    # teorik (1+d)^0.5 ile ondalık hanesine kadar tutuyor. Yani motor
    # SİSTEMATİK olarak ~%5 düşük değer üretiyordu.
    # ── B4-2: TERMİNAL REİNVESTMAN KISITI ─────────────────────────────
    # Terminal değer FCF'i g ile büyütüyor ama o büyümeyi finanse edecek
    # yatırımı hiç düşmüyordu — büyüme BEDAVA varsayılıyordu. McKinsey
    # değer-sürücüsü kimliği: g = ROIC × yeniden-yatırım oranı.
    # ÖLÇÜM: g=%2.5, ROIC=%10'da terminal akış %25 azalıyor (TV 1.33 kat
    # şişikti). roic_terminal verilmezse düzeltme UYGULANMAZ.
    yil = max(int(years), 1)
    hizli = max(1, yil // 2)
    orta_yil = bool(_ORTA_YIL)
    pv, fcf = 0.0, fcf0
    for t in range(1, yil + 1):
        if t <= hizli:
            g_t = growth
        else:
            pay = (t - hizli) / max(1, yil - hizli)
            g_t = growth + (terminal - growth) * pay
        fcf *= (1 + g_t)
        pv += fcf / (1 + discount) ** ((t - 0.5) if orta_yil else t)
    tv_akis = fcf * (1 + terminal)
    # O3: ×(1−g/RONIC) KALDIRILDI — FCF zaten yeniden-yatırımı düşer;
    # çarpan NOPAT tabanına aittir, burada çifte sayımdı. roic_terminal
    # bilgi/uyum amaçlı korunuyor.
    tv = tv_akis / (discount - terminal)
    pv += tv / (1 + discount) ** ((yil - 0.5) if orta_yil else yil)
    equity = pv - ((net_debt or 0.0) if adjust_net_debt else 0.0)
    return equity / shares


def implied_growth(price, fcf0, discount, terminal, years,
                   net_debt, shares, adjust_net_debt=True,
                   lo=-0.50, hi=0.60, roic_terminal=None):
    """
    TERS-DCF: Mevcut fiyatın İÇİNE GÖMÜLÜ yıllık FCF büyümesini çöz (bisection).
    Piyasanın ne beklediğini tahmin etmeyiz — fiyattan TÜRETİRİZ.
    Dönen: (büyüme, sınıra_çarptı_mı)
    """
    price, fcf0 = _num(price), _num(fcf0)
    if not price or price <= 0 or not fcf0 or fcf0 <= 0:
        return None, False
    f = lambda g: dcf_fair_value(fcf0, g, discount, terminal, years,
                                 net_debt, shares, adjust_net_debt,
                                 roic_terminal=roic_terminal)
    vlo, vhi = f(lo), f(hi)
    if vlo is None or vhi is None:
        return None, False
    if price <= vlo:
        return lo, True
    if price >= vhi:
        return hi, True
    for _ in range(80):
        mid = (lo + hi) / 2
        v = f(mid)
        if v is None:
            return None, False
        if v < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, False


def sensitivity(fcf0, growth, discount, terminal, years,
                net_debt, shares, adjust_net_debt, roic_terminal=None):
    """İskonto × büyüme duyarlılık matrisi (adil değer/hisse)."""
    if not _num(fcf0) or fcf0 <= 0:
        return None
    discs = [round(discount + d, 4) for d in (-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03)]
    grows = [round(growth + g, 4) for g in (-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06)]
    discs = [d for d in discs if d > terminal + 0.005]
    z = [[dcf_fair_value(fcf0, g, d, terminal, years, net_debt, shares,
                         adjust_net_debt, roic_terminal=roic_terminal)
          for g in grows] for d in discs]
    return {"discs": discs, "grows": grows, "z": z}


def dupont(m):
    """
    DuPont 5-FAKTÖR: ROE'nin NEDENİNİ ayrıştır. ROE tek başına 'ne kadar iyi'
    der; DuPont 'NEDEN iyi' der — marj mı, varlık devri mi, kaldıraç mı
    sürüyor? İki şirket aynı ROE'yi bambaşka yollarla üretebilir.
    ROE = Vergi Yükü × Faiz Yükü × EBIT Marjı × Varlık Devri × Kaldıraç
    """
    inc, bal = m.get("_inc"), m.get("_bal")
    if inc is None or bal is None:
        return None
    rev = by_year(get_row(inc, ROWS["revenue"]))
    ni = by_year(get_row(inc, ROWS["net_income"]))
    ebit = by_year(get_row(inc, ROWS["ebit"]))
    ptx = by_year(get_row(inc, ROWS["pretax"]))
    ta = by_year(get_row(bal, ROWS["total_assets"]))
    eq = by_year(get_row(bal, ROWS["equity"]))
    if not (rev and ni and ta and eq):
        return None
    y = max(set(rev) & set(ni) & set(ta) & set(eq))
    R, N = rev.get(y), ni.get(y)
    # ── DENETİM BULGUSU R: BAZ TUTARSIZLIĞI ──────────────────────────
    # dupont DÖNEM SONU özsermaye/varlık kullanıyordu; compute_metrics ise
    # m["roe"]'yi 2 YILLIK ORTALAMA özsermayeyle hesaplıyor (CFA standardı:
    # akış kalemi olan net kâr, stok kalemi olan özsermayenin ORTALAMASINA
    # bölünür). Sonuç: arayüzde ROE metriği ile DuPont'un roe_kontrol'ü
    # yan yana duruyor ve BİRBİRİNİ TUTMUYOR (testte %10 sapma).
    # DÜZELTME: DuPont da ortalama bazı kullanır. Kimlik korunur:
    #   (N/R) × (R/TA_ort) × (TA_ort/Eq_ort) = N/Eq_ort = m["roe"]
    _ys = sorted(set(ta) & set(eq))
    _p = _ys[_ys.index(y) - 1] if (y in _ys and _ys.index(y) > 0) else None
    A = ((ta[y] + ta[_p]) / 2 if (_p and ta.get(_p)) else ta.get(y))
    E = ((eq[y] + eq[_p]) / 2 if (_p and eq.get(_p) and eq[_p] > 0)
         else eq.get(y))
    _baz = ("2Y ortalama varlık/özsermaye (m['roe'] ile aynı baz)"
            if _p else "dönem sonu (tek yıl var)")
    if not (R and A and E) or N is None or E <= 0:
        return None
    net_marj = N / R
    varlik_devri = R / A
    kaldirac = A / E
    roe_3f = net_marj * varlik_devri * kaldirac
    out = {"donem": int(y),
           "net_marj": round(net_marj, 4),
           "varlik_devri": round(varlik_devri, 3),
           "kaldirac": round(kaldirac, 3),
           "roe_kontrol": round(roe_3f, 4),
           "baz": _baz,
           "ayirim": "ROE = net marj × varlık devri × finansal kaldıraç"}
    # 5-faktör (varsa): vergi yükü × faiz yükü × EBIT marjı × devir × kaldıraç
    EB, PT = ebit.get(y), ptx.get(y)
    # B5-9: EBIT veya vergi öncesi kâr NEGATİFSE vergi/faiz yükü oranları
    # ekonomik anlamını kaybeder; işaretleri ters dönüp ÇARPIM doğru ROE'yi
    # verirken faktör-faktör yorum saçmalar. Sıfır koruması yetmez.
    if EB and PT and EB > 0 and PT > 0:
        out["bes_faktor"] = {
            "vergi_yuku": round(N / PT, 3) if PT else None,      # NI/PTX
            "faiz_yuku": round(PT / EB, 3) if EB else None,      # PTX/EBIT
            "ebit_marj": round(EB / R, 4),                       # EBIT/Rev
            "varlik_devri": round(varlik_devri, 3),
            "kaldirac": round(kaldirac, 3)}
    # yorum: sürücü kim?
    drivers = []
    if kaldirac > 3.0:
        drivers.append(f"yüksek kaldıraç ({kaldirac:.1f}x) — ROE'yi borç "
                       f"şişiriyor, kırılganlık sinyali")
    if net_marj > 0.20:
        drivers.append(f"güçlü net marj (%{net_marj*100:.0f}) — fiyatlama "
                       f"gücü/verimlilik")
    elif net_marj < 0.05 and net_marj > 0:
        drivers.append(f"ince marj (%{net_marj*100:.1f}) — hacim/devir işi")
    if varlik_devri > 1.5:
        drivers.append(f"yüksek varlık devri ({varlik_devri:.1f}x) — "
                       f"sermaye-hafif model")
    elif varlik_devri < 0.4:
        drivers.append(f"düşük varlık devri ({varlik_devri:.2f}x) — "
                       f"sermaye-ağır model")
    out["surucu_notu"] = drivers or ["dengeli profil, tek baskın sürücü yok"]
    return out


def geri_alim_kalitesi(sbc, buyback, mcap, net_seyreltme=None):
    """
    GERİ ALIM KALİTESİ — 'iade mi, maaş mı?'

    SORUN: capital_allocation tüm geri alımı 'hissedara dağıtım' sayıyor.
    Ama geri alımın SBC seyreltmesini TELAFİ eden kısmı hissedara iade
    DEĞİLDİR — çalışan tazminatının nakit maliyetidir. Şirket bir eliyle
    hisse veriyor, öbür eliyle piyasadan geri alıyor; hissedarın payı
    değişmiyor, kasadan para çıkıyor. Bu ayrım yapılmazsa 'dağıtım' ve
    Buffett testi olduğundan İYİ görünür (Damodaran/Mauboussin).

    NOT: Bu bir DEĞERLEME düzeltmesi değil, SERMAYE TAHSİSİ teşhisidir.
    Değerleme tarafında seyreltme zaten iki kez yakalanıyor (SBC FCF'ten
    düşülüyor + oranlarda seyreltilmiş hisse) ve üçüncü bir düzeltme
    çifte sayım olurdu.
    """
    sbc, bb, mcap = _num(sbc), _num(buyback), _num(mcap)
    if not mcap or mcap <= 0:
        return None
    sbc = abs(sbc) if sbc else 0.0
    bb = abs(bb) if bb else 0.0
    if sbc <= 0 and bb <= 0:
        return None
    telafi = min(bb, sbc)                      # SBC'yi karşılayan kısım
    gercek = max(0.0, bb - sbc)                # asıl hissedar iadesi
    acik = max(0.0, sbc - bb)                  # karşılanmayan seyreltme
    if bb > 0:
        pay = telafi / bb
        if pay >= 0.90:
            hkm = ("Geri alımın NEREDEYSE TAMAMI SBC seyreltmesini telafi "
                   "ediyor — bu hissedara iade DEĞİL, tazminatın nakit "
                   "maliyetidir. 'Geri alım yapıyor' diye olumlu sayma.")
        elif pay >= 0.60:
            hkm = ("Geri alımın büyük kısmı SBC'yi telafi ediyor; gerçek "
                   "iade göründüğünden belirgin küçük.")
        elif pay >= 0.25:
            hkm = "Geri alımın bir kısmı SBC'yi telafi ediyor, kalanı iade."
        else:
            hkm = ("Geri alım büyük ölçüde GERÇEK hissedar iadesi — SBC "
                   "telafisi küçük pay.")
    else:
        pay = None
        hkm = ("Geri alım yok ama SBC var — seyreltme telafi EDİLMİYOR, "
               "hissedar payı doğrudan eriyor.")
    return {"sbc_musd": round(sbc / 1e6, 1),
            "geri_alim_musd": round(bb / 1e6, 1),
            "sbc_telafisi_musd": round(telafi / 1e6, 1),
            "gercek_iade_musd": round(gercek / 1e6, 1),
            "karsilanmayan_seyreltme_musd": round(acik / 1e6, 1),
            "telafi_payi": round(pay, 3) if pay is not None else None,
            "brut_sbc_seyreltme": round(sbc / mcap, 4),
            "gercek_iade_getirisi": round(gercek / mcap, 4),
            "net_seyreltme_gozlenen": net_seyreltme,
            "hukum": hkm,
            "not": ("SBC piyasa fiyatından verildiği varsayımıyla "
                    "[TÜRETİLMİŞ]. Sermaye tahsisi teşhisidir; değerleme "
                    "düzeltmesi DEĞİLDİR.")}


def arge_omru(sector, industry):
    """
    Ar-Ge'nin EKONOMİK ömrü sektöre göre değişir (Damodaran'ın sektör
    tablosu): yazılımda bir sürüm 2-3 yılda eskir, ilaçta bir molekül
    on yıl gelir üretir. Yanlış ömür seçmek düzeltmeyi de yanlış yapar.
    """
    s = (sector or "").lower()
    i = (industry or "").lower()
    if any(k in i for k in ("software", "internet", "saas", "application")):
        return 3
    if any(k in i for k in ("drug", "pharma", "biotech", "life science")):
        return 10
    if any(k in i for k in ("semiconductor", "hardware", "electronic",
                            "equipment")) or "technology" in s:
        return 5
    return 5


def roic_arge_kapitalize(rnd, ebit, ic, eff_tax, omur=5):
    """
    AR-GE KAPİTALİZASYONLU ROIC (Damodaran).

    SORUN: Muhasebe Ar-Ge'yi GİDER yazar, ama Ar-Ge ekonomik olarak
    YATIRIMDIR. Bu, ROIC'i İKİ yerden birden çarpıtır:
      • Pay (NOPAT): cari Ar-Ge'nin tamamı kârdan düşülmüş → kâr düşük
      • Payda (yatırılan sermaye): birikmiş Ar-Ge bilançoda YOK → sermaye
        çok daha düşük
    Payda daha sert küçüldüğü için NET ETKİ: ROIC ŞİŞER. Ar-Ge-yoğun
    şirketlerde (yazılım/ilaç/yarı iletken) bu şişme %10-15 puana ulaşır.

    DÜZELTME:
      Ar-Ge varlığı = Σ geçmiş Ar-Ge × (amorti edilmemiş kalan pay)
      Yıllık amortisman = Σ (geçmiş yılların Ar-Ge'si) / ömür
      Düzeltilmiş NOPAT = EBIT×(1−vergi) + (cari Ar-Ge − amortisman)
      Düzeltilmiş sermaye = yatırılan sermaye + Ar-Ge varlığı
    DİKKAT: (cari Ar-Ge − amortisman) farkı vergi-sonrası NOPAT'a ÖN-VERGİ
    değeriyle eklenir. Sebep: şirket Ar-Ge'yi gider yazıp vergi indirimini
    fiilen aldı; kapitalize ederken bu avantaj geri eklenince (1−vergi)
    denklemden düşer (Damodaran, Investment Valuation Böl. 9).

    Seri kısaysa ömür eldeki yıl sayısına kısılır [TÜRETİLMİŞ ~].
    Dönen: dict ya da None (veri yetersizse).
    """
    if not rnd or ebit is None or not ic or ic <= 0:
        return None
    ys = sorted(rnd)[-(int(omur) + 1):]
    L = len(ys)
    if L < 2:
        return None
    n_eff = L - 1                       # etkin amortisman ömrü
    varlik, amort = 0.0, 0.0
    for j, y in enumerate(reversed(ys)):        # j=0 → cari yıl
        v = _num(rnd.get(y))
        if v is None or v <= 0:
            continue
        if j == 0:
            varlik += v                          # cari yıl hiç amorti olmadı
        else:
            varlik += v * max(0, n_eff - j) / n_eff
            amort += v / n_eff
    cari = _num(rnd.get(ys[-1])) or 0.0
    if varlik <= 0:
        return None
    ic_adj = ic + varlik
    # ── VERGİ ETKİSİ (kritik incelik) ─────────────────────────────────
    # HATALI OLAN: nopat_adj = (EBIT + Ar-Ge − amortisman) × (1−vergi)
    # Bu, Ar-Ge düzeltmesini vergiden SONRA uygular. Ama şirket Ar-Ge'yi
    # zaten gider yazıp vergi indirimini FİİLEN ALDI — vergi faturası
    # değişmiyor. Damodaran (Investment Valuation, Bölüm 9) bu farklı
    # vergi avantajını geri ekler ve (1−vergi) denklemden DÜŞER:
    #     Düzeltilmiş NOPAT = EBIT×(1−vergi) + (cari Ar-Ge − amortisman)
    # Doğrulama (Damodaran'ın Amgen örneği): faaliyet kârındaki ön-vergi
    # artış ile vergi-sonrası artış BİRBİRİNE EŞİTTİR (701 mn = 701 mn).
    # Eski hâli düzeltmeyi (Ar-Ge − amortisman) × vergi kadar eksik
    # bırakıyordu; yani ROIC'i olması gerekenden DÜŞÜK gösteriyordu.
    _brut_duzeltme = cari - amort
    nopat_ham = ebit * (1 - eff_tax)
    nopat_adj = nopat_ham + _brut_duzeltme
    ham = nopat_ham / ic
    duz = nopat_adj / ic_adj
    return {"roic_ham": round(ham, 4),
            "roic_duzeltmeli": round(duz, 4),
            "fark_puan": round((duz - ham) * 100, 1),
            "arge_varligi_musd": round(varlik / 1e6, 1),
            "yillik_amortisman_musd": round(amort / 1e6, 1),
            "cari_arge_musd": round(cari / 1e6, 1),
            "nopat_ham_musd": round(nopat_ham / 1e6, 1),
            "nopat_duzeltmeli_musd": round(nopat_adj / 1e6, 1),
            "nopat_brut_duzeltme_musd": round(_brut_duzeltme / 1e6, 1),
            "omur_yil": int(omur), "etkin_omur_yil": n_eff,
            "kullanilan_yil": L,
            "omur_notu": (
                f"Sektör için HEDEFLENEN ömür {int(omur)} yıl; elde "
                f"{L} yıllık Ar-Ge verisi olduğu için ETKİN ömür {n_eff} "
                f"yıla düştü."
                + ("" if n_eff >= int(omur) else
                   " B5-8 DÜZELTMESİ: kısa seri Ar-Ge VARLIĞINI küçültür, "
                   "dolayısıyla yatırılan sermaye azalır ve düzeltilmiş ROIC "
                   "ham değere YAKLAŞIR — yani düzeltmenin etkisi OLDUĞUNDAN "
                   "ZAYIF görünür. (Eski not bunun TERSİNİ söylüyordu; "
                   "ölçüm: ömür 5→2 iken ROIC %17.17→%18.37 YÜKSELİYOR.) "
                   "Bu bir tercih değil VERİ KISITIDIR; daha uzun Ar-Ge "
                   "serisi (SEC 10-K) düzeltmeyi güçlendirir.")),
            "not": (f"Ar-Ge {n_eff} yıllık doğrusal amortismanla "
                    f"aktifleştirildi (Damodaran; hedef ömür {int(omur)}y). "
                    f"Ham ROIC ile makas büyükse muhasebe, işin gerçek "
                    f"sermaye verimini saklıyor demektir [TÜRETİLMİŞ ~].")}


def bakim_capex(rev, capex, ppe, dep):
    """
    BAKIM CAPEX AYRIMI (Greenwald yöntemi).

    SORUN: capex'in iki türü vardır ve muhasebe bunları ayırmaz:
      • BAKIM capex — mevcut işi ayakta tutmak; ZORUNLU
      • BÜYÜME capex — yeni kapasite; İSTEĞE BAĞLI
    İkisini toplam sayınca hızlı büyüyen şirket olduğundan KÖTÜ,
    sermaye-yoğun şirket olduğundan İYİ görünür (Buffett 1986,
    'owner earnings').

    YÖNTEM (Greenwald):
      1) PP&E/Satış oranının son yılların ortalaması alınır
      2) Büyüme capex ≈ ΔSatış × bu oran
      3) Bakım capex = toplam capex − büyüme capex  ([0, toplam] aralığına
         kıstırılır)
      4) Satış DÜŞTÜYSE büyüme capex'i yoktur → hepsi bakımdır

    DİKKAT: Bu ayrım EPV (büyümesiz taban değer) içindir. Büyüme
    projeksiyonlu DCF'te bakım capex kullanmak ÇİFTE SAYIMDIR — çünkü
    projelediğin büyümeyi üreten şey tam da büyüme capex'idir. Bu yüzden
    ana DCF'in FCF bazı DEĞİŞTİRİLMEZ.
    """
    rev, capex = rev or {}, capex or {}
    ppe, dep = ppe or {}, dep or {}
    ortak = sorted(set(rev) & set(ppe))
    if len(ortak) < 2 or not capex:
        return None
    oranlar = [ppe[y] / rev[y] for y in ortak[-5:] if rev.get(y)]
    if not oranlar:
        return None
    oran = sum(oranlar) / len(oranlar)
    ys = sorted(set(rev) & set(capex))
    if len(ys) < 2:
        return None
    t, p = ys[-1], ys[-2]
    toplam = abs(_num(capex.get(t)) or 0.0)
    if toplam <= 0:
        return None
    d_satis = (rev.get(t) or 0) - (rev.get(p) or 0)
    if d_satis <= 0:
        buyume, gerekce = 0.0, "satış daraldı → büyüme capex'i yok"
    else:
        buyume = min(max(d_satis * oran, 0.0), toplam)
        gerekce = f"ΔSatış × ort. PP&E/Satış ({oran:.2f})"
    bakim = max(0.0, toplam - buyume)
    d0 = abs(_num(dep.get(t)) or 0.0)
    return {"donem": int(t),
            "toplam_capex_musd": round(toplam / 1e6, 1),
            "bakim_capex_musd": round(bakim / 1e6, 1),
            "buyume_capex_musd": round(buyume / 1e6, 1),
            "buyume_payi": round(buyume / toplam, 3),
            "ppe_satis_orani": round(oran, 3),
            "amortisman_musd": round(d0 / 1e6, 1) if d0 else None,
            "amortisman_bakim_farki_musd": (round((d0 - bakim) / 1e6, 1)
                                            if d0 else None),
            "gerekce": gerekce,
            "not": ("Greenwald ayrımı [TÜRETİLMİŞ]. Büyüme payı yüksekse "
                    "şirket geleceğe yatırım yapıyor ve raporlanan FCF "
                    "gerçek nakit üretme gücünü OLDUĞUNDAN DÜŞÜK gösterir. "
                    "Bakım capex amortismandan büyükse tersi geçerlidir: "
                    "şirket göründüğünden daha çok nakit yakıyor. Bu ayrım "
                    "EPV'de kullanılır; büyüme projeksiyonlu DCF'te "
                    "kullanmak çifte sayım olurdu.")}


def epv_value(m, a):
    """
    EPV (Earnings Power Value — Greenwald): BÜYÜMESİZ sürdürülebilir kazanç
    gücü. EPV = düzeltilmiş NOPAT / WACC. DCF 'iyimser senaryo' ise EPV
    'taban'dır — büyümeye hiç para ödemeden şirket ne eder? Fiyat EPV'nin
    altındaysa piyasa küçülme fiyatlıyor; çok üstündeyse tüm değer büyüme
    beklentisinde, yani kırılgan. 'Kaliteye fazla ödeme' tuzağına çıpa.
    """
    inc = m.get("_inc")
    if inc is None:
        return None
    ebit = by_year(get_row(inc, ROWS["ebit"]))
    if not ebit:
        return None
    ys = sorted(ebit)
    # normalize EBIT: son 3 yıl ortalaması (tek yıl sivriliğini yumuşat)
    son = [ebit[y] for y in ys[-3:] if ebit[y] is not None]
    if not son:
        return None
    ebit_norm = sum(son) / len(son)
    if ebit_norm <= 0:
        return {"uygulanamaz": "Normalize EBIT negatif — EPV anlamsız "
                               "(şirket sürdürülebilir kâr üretmiyor)."}
    # GREENWALD BAKIM-CAPEX DÜZELTMESİ: EBIT, amortismanı zaten düşmüştür.
    # Sürdürülebilir kazanç gücü için doğru gider AMORTİSMAN değil BAKIM
    # CAPEX'idir. Amortisman bakım capex'inden büyükse fark vergi sonrası
    # GERİ EKLENİR (şirket göründüğünden çok nakit üretiyor); küçükse
    # DÜŞÜLÜR (varlıklar tükeniyor). Eskiden bu "yaklaşık" diye atlanıyordu.
    _bc = m.get("capex_ayrim") or {}
    _duz = _num(_bc.get("amortisman_bakim_farki_musd"))
    _bc_not = "bakım-capex düzeltmesi yok (veri yetersiz)"
    if _duz is not None:
        ebit_norm = ebit_norm + _duz * 1e6
        _bc_not = (f"amortisman − bakım capex = {_duz:+,.0f}M USD "
                   f"normalize EBIT'e eklendi (Greenwald)")
        if ebit_norm <= 0:
            return {"uygulanamaz": "Bakım-capex düzeltmesi sonrası "
                                   "normalize EBIT negatif — şirket bakım "
                                   "yatırımını kazancından karşılayamıyor."}
    nopat = ebit_norm * (1 - m.get("eff_tax", 0.21))
    wacc = a["discount"]
    ev_epv = nopat / wacc
    # özsermaye değeri: EV − net borç; hisse başı
    eq_val = ev_epv - (m.get("net_debt") or 0)
    shares = m.get("shares")
    epv_ps = eq_val / shares if shares else None
    price = m.get("price")
    return {"epv_hisse_basi": round(epv_ps, 2) if epv_ps else None,
            "normalize_ebit_musd": round(ebit_norm / 1e6, 1),
            "nopat_musd": round(nopat / 1e6, 1),
            "wacc": round(wacc, 4),
            "fiyata_gore": (round(price / epv_ps - 1, 3)
                            if (epv_ps and price and epv_ps > 0) else None),
            "yorum": ("Büyümesiz taban değer. Fiyat bunun ALTINDAysa piyasa "
                      "küçülme fiyatlıyor (derin değer ya da tuzak); ÇOK "
                      "ÜSTÜNDEyse tüm değer büyüme beklentisinde, kırılgan."),
            "bakim_capex_duzeltmesi": _bc_not,
            "capex_ayrim": _bc or None,
            "not": "3Y ortalama EBIT ile normalize + Greenwald bakım-capex "
                   "düzeltmesi (varsa). İskonto: CAPM özsermaye maliyeti "
                   "(Greenwald orijinali WACC kullanır; kaldıraçlı "
                   "şirkette özsermaye maliyeti ≥ WACC olduğundan bu "
                   "sürüm DAHA TEMKİNLİDİR — rapor G6)."}


def cash_cycle(m):
    """Nakit Dönüşüm Döngüsü = DIO + DSO − DPO. Negatif = tedarikçi parasıyla
    bedava finansman (Apple/Amazon). Artan CCC = işletme sermayesi stresi."""
    inc, bal = m.get("_inc"), m.get("_bal")
    if inc is None or bal is None:
        return None
    rev = by_year(get_row(inc, ROWS["revenue"]))
    cogs = by_year(get_row(inc, ROWS["cogs"]))
    recv = by_year(get_row(bal, ROWS["receivables"]))
    inv = by_year(get_row(bal, ROWS["inventory"]))
    pay = by_year(get_row(bal, ROWS["payables"]))
    if not (rev and recv):
        return None

    def ccc_for(y):
        R, C = rev.get(y), cogs.get(y)
        dso = (recv.get(y, 0) / R * 365) if R else None
        dio = (inv.get(y, 0) / C * 365) if (C and inv.get(y) is not None) else None
        dpo = (pay.get(y, 0) / C * 365) if (C and pay.get(y) is not None) else None
        if dso is None:
            return None
        ccc = dso + (dio or 0) - (dpo or 0)
        return {"dso": round(dso, 1),
                "dio": round(dio, 1) if dio is not None else None,
                "dpo": round(dpo, 1) if dpo is not None else None,
                "ccc": round(ccc, 1)}
    ys = sorted(rev)
    seri = {int(y): ccc_for(y) for y in ys[-3:] if ccc_for(y)}
    if not seri:
        return None
    yr = sorted(seri)
    trend = None
    if len(yr) >= 2:
        d = seri[yr[-1]]["ccc"] - seri[yr[0]]["ccc"]
        trend = ("artıyor (işletme sermayesi stresi olabilir)" if d > 10
                 else "iyileşiyor (nakit hızlanıyor)" if d < -10
                 else "stabil")
    return {"seri": seri, "trend": trend,
            "not": "CCC = DIO + DSO − DPO. Negatif = tedarikçi finansmanı "
                   "(güç); artan = büyümenin nakit maliyeti."}


def scenario_ev(m):
    """
    SENARYO-OLASILIK AĞIRLIKLI BEKLENEN DEĞER (Mauboussin/Rappaport).
    Monte Carlo dağılımını 3 ayrık senaryoya indir (ayı/baz/boğa), olasılık
    ata, beklenen değeri hesapla, fiyatla karşılaştır. 'Priced for
    perfection' (mükemmelliğe fiyatlanmış) uyarısı üretir.
    """
    mc = m.get("mc")
    price = m.get("price")
    if not mc or not price:
        return None
    p25, p50, p75 = mc.get("p25"), mc.get("p50"), mc.get("p75")
    if None in (p25, p50, p75):
        return None
    # ── B6-6 ─────────────────────────────────────────────────────────
    # Çeyreklikler bölgelerin ORTALAMASI değil SINIRIDIR; ayrıca P25/P50/P75
    # tanım gereği dört EŞİT %25'lik parça üretir — onları 30/45/25 ile
    # ağırlıklamak kendi içinde tutarsızdır. ÖLÇÜM (200.000 örnek): sağa
    # çarpık dağılımda gerçek ortalama 104.66 iken eski yöntem 100.20
    # (−%4.3); KOŞULLU ORTALAMA + 25/50/25 ise TAM (%0.00 sapma).
    _dag = mc.get("dagilim") or []
    if len(_dag) >= 20:
        _a = np.array(_dag, dtype=float)
        ayi = float(_a[_a <= p25].mean()) if (_a <= p25).any() else p25
        baz = (float(_a[(_a > p25) & (_a < p75)].mean())
               if ((_a > p25) & (_a < p75)).any() else p50)
        boga = float(_a[_a >= p75].mean()) if (_a >= p75).any() else p75
        w = {"ayi": 0.25, "baz": 0.50, "boga": 0.25}
    else:
        ayi, baz, boga = p25, p50, p75
        w = {"ayi": 0.30, "baz": 0.45, "boga": 0.25}
    ev = ayi * w["ayi"] + baz * w["baz"] + boga * w["boga"]
    getiri = ev / price - 1
    # asimetri: yukarı alan / aşağı alan
    yukari = (boga - price) / price
    asagi = (price - ayi) / price
    asimetri = (yukari / asagi) if asagi > 0.01 else None
    if getiri < -0.15:
        hkm = ("Beklenen değer fiyatın belirgin ALTINDA — olasılık ağırlıklı "
               "olarak fiyat pahalı; mükemmelliğe yakın icra fiyatlanmış.")
    elif getiri > 0.15:
        hkm = ("Beklenen değer fiyatın belirgin ÜSTÜNDE — olasılık ağırlıklı "
               "olarak cazip; ama önce senaryo olasılıklarını sorgula.")
    else:
        hkm = ("Beklenen değer fiyata yakın — asimetriye ve katalizöre bak, "
               "kenar değerlemeden gelmiyor.")
    return {"senaryolar": {
                "ayi": {"deger": ayi, "olasilik": w["ayi"]},
                "baz": {"deger": baz, "olasilik": w["baz"]},
                "boga": {"deger": boga, "olasilik": w["boga"]}},
            "beklenen_deger": round(ev, 2),
            "fiyat": round(price, 2),
            "beklenen_getiri": round(getiri, 3),
            "yukari_asagi_asimetri": (round(asimetri, 2)
                                      if asimetri else None),
            "hukum": hkm,
            # Denetim Bulgu 6b: eski not sabit "30/45/25" diyordu ama ana
            # yol (B6-6) koşullu-ortalama + 25/50/25 kullanıyor — metin
            # fiilen kullanılan değerlerden üretilir, yalan söyleyemez.
            "not": (f"Senaryolar Monte Carlo dağılımından "
                    + ("(koşullu-ortalama yöntemi — B6-6)"
                       if len(_dag) >= 20 else
                       "(çeyreklik-sınır yedeği — dağılım kısa)")
                    + f"; olasılıklar ayı %{w['ayi']*100:.0f} / "
                      f"baz %{w['baz']*100:.0f} / "
                      f"boğa %{w['boga']*100:.0f}. Claude bunları "
                      f"kendi analiziyle revize edebilir.")}


def cyclical_trap(m):
    """
    ÇEVRİMSEL ÇARPAN TUZAĞI + değer tuzağı uyarıları. Çevrimsel işlerde
    (madencilik, yarı iletken, enerji, otomotiv, kimya) TEK yıl kazancı
    yanıltıcıdır: tepe kazançta düşük F/K = TUZAK, dip kazançta yüksek F/K =
    genelde fırsat. Marj tepedeyse + F/K düşükse uyar.
    """
    sektor = (m.get("sector") or "").lower()
    ind = (m.get("industry") or "").lower()
    cyc_kw = ("cyclical", "semiconductor", "energy", "oil", "gas", "mining",
              "metals", "materials", "automobile", "auto", "chemical",
              "basic material", "steel")
    is_cyclical = any(k in sektor or k in ind for k in cyc_kw)
    notes = []
    om = m.get("om") or {}          # faaliyet marjı serisi
    pe = m.get("pe_t") or m.get("pe_f")
    if is_cyclical and om and len(om) >= 3:
        ys = sorted(om)
        # DENETİM BULGUSU X: ORTALAMA yerine MEDYAN. Çevrimsel bir seride
        # tek bir zirve yılı ortalamayı yukarı çeker (testte %40) ve
        # "zirve tuzağı" testinin eşiğini yükselterek TUZAĞI GİZLER.
        # normalized_earnings zaten medyan kullanıyordu — iki motor
        # çelişiyordu; ikisi de medyana geçti.
        cur, avg = om[ys[-1]], _medyan(om.values())
        if cur > avg * 1.25 and pe and pe < 15:
            notes.append(
                f"ÇEVRİMSEL ZİRVE TUZAĞI: faaliyet marjı (%{cur*100:.0f}) "
                f"tarihsel MEDYANIN (%{avg*100:.0f}) belirgin üstünde ve "
                f"F/K düşük ({pe:.0f}x) — bu 'ucuzluk' aldatıcı olabilir; "
                f"tepe kazançta düşük F/K çevrimsel tuzaktır. Normalize "
                f"(mid-cycle) kazançla bak.")
        elif cur < avg * 0.75 and pe and pe > 25:
            notes.append(
                f"ÇEVRİMSEL DİP: marj (%{cur*100:.0f}) medyanın altında, "
                f"F/K yüksek — çevrimsel işlerde bu genelde tuzak DEĞİL, "
                f"toparlanma öncesi giriş bölgesi olabilir.")
    # değer tuzağı: skor ucuz ama kalite düşük + revizyon aşağı
    score, qual = m.get("score"), m.get("quality")
    rv = (m.get("revizyon") or {}).get("yon")
    if score is not None and score < 35 and qual is not None and qual < 45:
        extra = " ve analist revizyonları aşağı" if rv == "AŞAĞI" else ""
        notes.append(
            f"DEĞER TUZAĞI RİSKİ: fiyat ucuz görünüyor (skor {score:.0f}) "
            f"ama kalite düşük (skor {qual:.0f}){extra} — ucuzluğun bir "
            f"NEDENİ olabilir; 'neden bu kadar ucuz?' sorusu cevaplanmadan "
            f"cazibe sayma.")
    return {"cevrimsel_mi": is_cyclical, "uyarilar": notes or None}


def sector_template(m):
    """
    SEKTÖR ŞABLONU: Genel metrikler her sektörde aynı hikâyeyi anlatmaz.
    SaaS'ta Rule of 40, bankada ROTCE/P-TBV, çevrimselde normalize F/K
    doğru mercek. Hesaplanabilenler burada; hesaplanamayanlar Claude'a
    açık web görevi olarak devredilir (dürüstlük: uydurma yok).
    """
    sec = (m.get("sector") or "").lower()
    ind = (m.get("industry") or "").lower()
    rev = m.get("rev") or {}
    ys = sorted(rev)
    out = None

    def son_buyume():
        if len(ys) >= 2 and rev.get(ys[-2]):
            return rev[ys[-1]] / rev[ys[-2]] - 1
        return m.get("rev_cagr3")

    if "software" in ind or "saas" in ind:
        g = son_buyume()
        fcfm = (m["fcf_base"] / rev[ys[-1]]
                if (m.get("fcf_base") and ys and rev.get(ys[-1])) else None)
        r40 = (g + fcfm) if (g is not None and fcfm is not None) else None
        out = {"tur": "SaaS/Yazılım",
               "metrikler": {"gelir_buyume": round(g, 3) if g is not None else None,
                             "fcf_marji": round(fcfm, 3) if fcfm is not None else None,
                             "rule_of_40": round(r40, 3) if r40 is not None else None},
               "yorum": (None if r40 is None else
                         ("Rule of 40 GEÇTİ (%{:.0f}) — büyüme/kârlılık "
                          "dengesi sağlıklı.".format(r40 * 100) if r40 >= 0.40
                          else "Rule of 40 sınırda (%{:.0f}).".format(r40 * 100)
                          if r40 >= 0.25 else
                          "Rule of 40 ALTINDA (%{:.0f}) — ne yeterince "
                          "büyüyor ne kâr ediyor.".format(r40 * 100))),
               "claude_gorevleri": ["NRR/net dollar retention rakamını son "
                                    "çağrı/10-K'dan webden bul (pakette yok)."]}

    elif "bank" in ind:
        inc, bal = m.get("_inc"), m.get("_bal")
        eq = by_year(get_row(bal, ROWS["equity"])) if bal is not None else {}
        gw = by_year(get_row(bal, ROWS["goodwill"])) if bal is not None else {}
        oi_ = by_year(get_row(bal, ROWS["oth_intan"])) if bal is not None else {}
        gwi = by_year(get_row(bal, ROWS["gw_intan"])) if bal is not None else {}
        ni = by_year(get_row(inc, ROWS["net_income"])) if inc is not None else {}
        y = max(eq) if eq else None
        tang = None
        if y and eq.get(y):
            ded = (gw.get(y, 0) + oi_.get(y, 0)) or gwi.get(y, 0)
            tang = eq[y] - ded
        rotce = (ni.get(y) / tang) if (y and ni.get(y) and tang and tang > 0) else None
        sh = m.get("shares")
        tbv_ps = tang / sh if (tang and sh) else None
        ptbv = (m["price"] / tbv_ps) if (tbv_ps and m.get("price")) else None
        kimlik = None
        if ptbv and rotce and m.get("pe_t"):
            beklenen = m["pe_t"] * rotce
            kimlik = {"p_tbv": round(ptbv, 2),
                      "fk_x_rotce": round(beklenen, 2),
                      "not": "P/TBV ≈ F/K × ROTCE kimliği; büyük sapma, "
                             "F/K veya defter kaleminde tutarsızlık demektir."}
        out = {"tur": "Banka",
               "metrikler": {"rotce": round(rotce, 4) if rotce else None,
                             "tbv_hisse": round(tbv_ps, 2) if tbv_ps else None,
                             "p_tbv": round(ptbv, 2) if ptbv else None},
               "kimlik_kontrol": kimlik,
               "yorum": ("Bankada ROE yerine ROTCE, F/K yerine P/TBV esas "
                         "mercektir; DCF/EPV bankada güvenilmezdir."),
               "claude_gorevleri": ["NIM (net faiz marjı), kredi/mevduat ve "
                                    "takipteki kredi oranını 10-K/10-Q'dan "
                                    "webden çıkar."]}

    elif any(k in ind or k in sec for k in
             ("semiconductor", "energy", "oil", "mining", "metals", "steel",
              "chemical", "auto")):
        om = m.get("om") or {}
        if len(om) >= 3:
            # BULGU X: mid-cycle marj MEDYAN olmalı — tek zirve yılı
            # ortalamayı bozar ve normalize F/K'yı düşük gösterir.
            om_avg = _medyan(om.values())
            dil = m.get("dil_shares_seri") or {}
            sh = dil.get(max(dil)) if dil else m.get("shares")
            _eff2 = m.get("eff_tax", 0.21)
            _fz2 = m.get("int_exp")   # Y1: faiz-sonrası taban
            neps = ((rev[ys[-1]] * om_avg * (1 - _eff2)
                     - (abs(_fz2) * (1 - _eff2) if _fz2 else 0.0)) / sh
                    if (ys and rev.get(ys[-1]) and sh) else None)
            npe = (m["price"] / neps) if (neps and neps > 0 and m.get("price")) else None
            out = {"tur": "Çevrimsel (mid-cycle mercek)",
                   "metrikler": {
                       "midcycle_faaliyet_marji_medyan": round(om_avg, 4),
                       "guncel_marj": round(om[max(om)], 4),
                       "normalize_eps": round(neps, 2) if neps else None,
                       "normalize_fk": round(npe, 1) if npe else None,
                       "raporlanan_fk": m.get("pe_t"),
                       "normalize_baz": ("faiz-sonrası" if _fz2 else
                                         "kaldıraçsız [faiz yok]")},
                   "yorum": ("Çevrimselde GERÇEK F/K normalize olandır: "
                             "raporlanan F/K düşük ama normalize F/K yüksekse "
                             "'ucuzluk' zirve kazancının yanılsamasıdır."),
                   "claude_gorevleri": ["Döngünün neresindeyiz? Fiyat/talep "
                                        "göstergelerini (ör. bellek fiyatı, "
                                        "emtia fiyatı, book-to-bill) webden "
                                        "kontrol et."],
                   "not": "Mid-cycle, eldeki ~4 yıllık marj ortalaması — "
                          "tam döngü (7-10y) değil [TÜRETİLMİŞ]."}

    elif any(k in ind for k in ("drug", "pharma", "biotech")):
        rr = m.get("rnd_rev") or {}
        out = {"tur": "İlaç/Biyoteknoloji",
               "metrikler": {"arge_gelir": (round(rr[max(rr)], 3)
                                            if rr else None)},
               "yorum": ("İlaçta değer ürün-bazlıdır: tek ilaç DCF'i "
                         "yanıltır; patent uçurumu ilk yılda gelirin "
                         "%60-80'ini götürebilir."),
               "claude_gorevleri": ["En büyük 2-3 ilacın gelir payı ve patent "
                                    "bitiş tarihlerini webden çıkar; yaklaşan "
                                    "uçurum varsa katalizör haritasına koy.",
                                    "Faz-3 pipeline'ı listele."]}

    elif "insurance" in ind:
        out = {"tur": "Sigorta",
               "metrikler": {},
               "yorum": "Sigortada combined ratio ve float getirisi esastır; "
                        "standart FCF/DCF yanıltır.",
               "claude_gorevleri": ["Combined ratio ve float büyümesini "
                                    "10-K'dan webden çıkar."]}
    elif "reit" in ind:
        out = {"tur": "REIT",
               "metrikler": {},
               "yorum": "REIT'te net kâr değil FFO/AFFO esastır; temettü "
                        "sürdürülebilirliği FFO'ya göre ölçülür.",
               "claude_gorevleri": ["FFO/AFFO ve payout oranını webden bul."]}
    return out


def capital_allocation(m):
    """
    SERMAYE TAHSİSİ SKORU (Thorndike/Outsiders çerçevesi): CEO kazanılan
    parayı nereye koyuyor ve bu değer YARATIYOR mu? Bileşenler: ROIC seviye+
    trend, Buffett testi (tutulan kâr → piyasa değeri), dağıtımın FCF'ten
    karşılanması, dilüsyon, satın alma yoğunluğu. 0-100; yaklaşık — Mcap
    tarihi yıllık ortalama fiyat×bugünkü hisse ile TÜRETİLMİŞtir.
    """
    inc, bal, cf = m.get("_inc"), m.get("_bal"), m.get("_cf")
    hist = m.get("_hist")
    if inc is None or cf is None:
        return None
    ebit = by_year(get_row(inc, ROWS["ebit"]))
    tax = by_year(get_row(inc, ROWS["tax"]))
    ptx = by_year(get_row(inc, ROWS["pretax"]))
    ni = by_year(get_row(inc, ROWS["net_income"]))
    fcf = by_year(get_row(cf, ROWS["fcf"]))
    div = by_year(get_row(cf, ROWS["div_paid"]))
    bb = by_year(get_row(cf, ROWS["buyback"]))
    re_ = by_year(get_row(bal, ROWS["retained"])) if bal is not None else {}
    gw = by_year(get_row(bal, ROWS["goodwill"])) if bal is not None else {}
    eq = by_year(get_row(bal, ROWS["equity"])) if bal is not None else {}
    icap = by_year(get_row(bal, ROWS["invested_cap"])) if bal is not None else {}

    detay, skor = {}, 50.0

    # 1) ROIC serisi + trend
    roic_seri = {}
    for y in sorted(set(ebit) & set(icap)):
        t = (tax.get(y) / ptx.get(y)
             if (tax.get(y) is not None and ptx.get(y)) else None)
        t = t if (t is not None and 0 <= t <= 0.45) else m.get("eff_tax", 0.21)
        prev = icap.get(y - 1)
        ic = (icap[y] + prev) / 2 if prev else icap[y]
        if ic and ic > 0 and ebit.get(y) is not None:
            roic_seri[int(y)] = round(ebit[y] * (1 - t) / ic, 4)
    detay["roic_seri"] = roic_seri or None
    if roic_seri:
        yr = sorted(roic_seri)
        son = roic_seri[yr[-1]]
        if son > 0.15:
            skor += 15
        elif son > 0.08:
            skor += 5
        elif son < 0.05:
            skor -= 10
        if len(yr) >= 3:
            if son >= roic_seri[yr[0]] + 0.01:
                skor += 8
                detay["roic_trend"] = "yükseliyor"
            elif son <= roic_seri[yr[0]] - 0.02:
                skor -= 8
                detay["roic_trend"] = "düşüyor"
            else:
                detay["roic_trend"] = "stabil"

    # 2) Buffett testi: ΔMcap / tutulan kâr (yaklaşık)
    try:
        shares = m.get("shares")
        if hist is not None and len(hist) and shares and ni:
            close = hist["Close"].dropna()
            yr_all = sorted(set(ni))
            span = [y for y in yr_all if (close.index.year == y).any()]
            if len(span) >= 3:
                y0, y1 = span[0], span[-1]
                # ── B6-1 (KRİTİK) ────────────────────────────────
                # Tarihsel piyasa değeri "o yılın ort. fiyatı × BUGÜNKÜ
                # hisse adedi" ile hesaplanıyordu. Geri alım yapan şirkette
                # bugünkü adet DÜŞÜK olduğu için BAŞLANGIÇ mcap'i küçük
                # görünüyor → ΔMcap şişiyor → oran YUKARI sapıyor.
                # ÖLÇÜM: %30 geri alımda gerçek 0.80 → kod 1.40 (+%75) ve
                # puan +3'ten +12'ye sıçrıyor. Yani skor, tam da ödüllendirmek
                # istediği "iyi sermaye tahsisçisi"ni HAKSIZ YERE kayırıyordu.
                _sh_seri = m.get("dil_shares_seri") or {}

                def _adet(yil):
                    if _sh_seri.get(yil):
                        return _sh_seri[yil]
                    if _sh_seri:
                        _yak = sorted(_sh_seri,
                                      key=lambda yy: abs(yy - yil))
                        return _sh_seri.get(_yak[0]) or shares
                    return shares

                p0 = float(close[close.index.year == y0].mean())
                p1 = float(close[close.index.year == y1].mean())
                if p0 != p0 or p1 != p1:      # B-katmanı: yıl fiyatı yoksa
                    p0 = p1 = None            # NaN detaya sızmasın
                # B6-2: geri alımlar çoğu şirkette DOĞRUDAN retained
                # earnings'e borç kaydedilir → ΔRE tutulan kârı OLDUĞUNDAN
                # KÜÇÜK gösterir ve oranın PAYDASI küçülür; bu B6-1 ile AYNI
                # YÖNDE ikinci bir yanlılıktır. Artık (ΣNI − Σtemettü)
                # BİRİNCİL, ΔRE yalnız çapraz kontrol.
                # NOT: nakit akışında temettü NEGATİF saklıdır → TOPLANIR.
                # Y2 — HİSSE BAŞI baz: şirket-düzeyi payda geri alımı
                # düşmüyordu; EPS−DPS tabanında adet etkisi otomatik doğru.
                tutulan = sum((ni.get(y, 0) + div.get(y, 0)) / _adet(y)
                              for y in span if _adet(y))
                _re_capraz = None
                if (re_ and re_.get(y0) is not None
                        and re_.get(y1) is not None):
                    _re_capraz = re_[y1] - re_[y0]
                if tutulan and tutulan > 0 and p0 is not None:
                    oran = (p1 - p0) / tutulan
                    detay["buffett_testi"] = {
                        "donem": f"{y0}→{y1}",
                        "deger_yaratma_orani": round(oran, 2),
                        "not": ("HİSSE BAŞI (Y2): Δ(yıllık ort. fiyat) / "
                                "Σ(EPS−DPS); ≥1 iyi — geri alım etkisi bu "
                                "tabanda otomatik doğru işlenir."),
                        "re_degisim_kontrol_musd": (
                            round(_re_capraz / 1e6, 0)
                            if _re_capraz is not None else None)}
                    if oran >= 1.5:
                        skor += 18
                    elif oran >= 1.0:
                        skor += 12
                    elif oran >= 0.5:
                        skor += 3
                    else:
                        skor -= 10
    except Exception:
        pass

    # 3) Dağıtım sürdürülebilirliği: (temettü+geri alım) vs FCF (3Y)
    ys3 = sorted(set(fcf))[-3:]
    if ys3:
        fcf_top = sum(fcf.get(y, 0) for y in ys3)
        dag = sum(abs(div.get(y, 0)) + abs(bb.get(y, 0)) for y in ys3)
        if dag > 0 and fcf_top > 0:
            kap = dag / fcf_top
            detay["dagitim"] = {
                "temettu_geri_alim_3y_musd": round(dag / 1e6, 0),
                "fcf_3y_musd": round(fcf_top / 1e6, 0),
                "fcf_karsilama": round(kap, 2)}
            _ga = m.get("geri_alim") or {}
            if _ga.get("telafi_payi") is not None:
                detay["dagitim"]["geri_alim_kalitesi"] = (
                    f"Geri alımın %{_ga['telafi_payi']*100:.0f}'i SBC "
                    f"telafisi; gerçek iade "
                    f"${_ga['gercek_iade_musd']:,.0f}M "
                    f"(getiri %{(_ga['gercek_iade_getirisi'] or 0)*100:.1f})")
            if kap <= 0.9:
                skor += 10
            elif kap > 1.2:
                skor -= 10
                detay["dagitim"]["uyari"] = ("Dağıtım FCF'i aşıyor — borç/"
                                             "nakitle finanse ediliyor "
                                             "olabilir.")

    # 4) Dilüsyon / geri alım yönü
    dil = m.get("dilution_yoy")
    if dil is not None:
        if dil < -0.005:
            skor += 8
            detay["hisse_yonu"] = "net geri alım (hisse azalıyor)"
        elif dil > 0.06:
            skor -= 10
            detay["hisse_yonu"] = "yoğun dilüsyon"

    # 5) Satın alma yoğunluğu: goodwill/özkaynak şişmesi
    if gw and eq:
        y = max(set(gw) & set(eq), default=None)
        y0 = min(set(gw) & set(eq), default=None)
        if y and eq.get(y) and gw.get(y) is not None:
            pay = gw[y] / eq[y]
            if pay > 0.5 and y0 and gw.get(y0) is not None and eq.get(y0) \
                    and (gw[y] - gw[y0]) > (eq[y] - eq[y0]):
                skor -= 8
                detay["satin_alma"] = (f"Şerefiye özkaynağın "
                                       f"%{pay*100:.0f}'i ve özkaynaktan "
                                       f"hızlı büyümüş — büyüme satın alma "
                                       f"ağırlıklı; entegrasyon/değer "
                                       f"yıkımı riski.")

    skor = float(max(5, min(95, skor)))
    if skor >= 70:
        profil = "DİSİPLİNLİ — sermaye değer yaratacak yerlere gidiyor"
    elif skor >= 45:
        profil = "KARIŞIK — güçlü ve zayıf sinyaller bir arada"
    else:
        profil = "SORUNLU — sermaye tahsisi değer yaratmıyor olabilir"
    return {"skor": round(skor, 0), "profil": profil, "detay": detay,
            "not": "Thorndike/Outsiders çerçevesi; Mcap tarihi yaklaşıktır "
                   "[TÜRETİLMİŞ]."}


def base_rates(m, a):
    """
    BAZ ORANLAR (dış görüş, Mauboussin tarzı): 'Benzer şirketler tarihsel
    olarak ne başardı?' Rakamlar literatür ÖZETİ ve YAKLAŞIKtır [TÜRETİLMİŞ]
    — kesin istatistik değil; amaç, içeriden bakışın (bu şirket özel)
    dışarıdan bakışla (taban oranlar) çarpışması.
    """
    aday = [a.get("growth")]
    if m.get("implied_g") is not None and not m.get("implied_capped"):
        aday.append(m["implied_g"])
    aday = [x for x in aday if x is not None]
    if not aday:
        return None
    hedef = max(aday)
    if hedef >= 0.25:
        oran = "~%1-3 (çok nadir — on yılda %25+ bileşik büyüme istisnadır)"
        mesaj = ("Fiyatın/varsayımın istediği büyüme, tarihsel olarak "
                 "şirketlerin yalnızca küçük bir azınlığında görüldü. "
                 "İstisna olduğu KANITLANMALI — varsayılamaz.")
    elif hedef >= 0.15:
        oran = "~%5-8 (nadir)"
        mesaj = ("İstenen büyüme nadir ama imkânsız değil; moat ve pazar "
                 "büyüklüğü kanıtı şart.")
    elif hedef >= 0.10:
        oran = "~%12-18 (az rastlanır)"
        mesaj = "Makul-iddialı bölge; rekabet avantajı kalıcılığına bak."
    else:
        oran = "yaygın (~%30+)"
        mesaj = "Mütevazı hedef — baz oran engel değil."
    cagr = m.get("rev_cagr3")
    kiyas = None
    if cagr is not None:
        kiyas = (f"Şirketin kendi 3Y CAGR'ı %{cagr*100:.0f} — hedef "
                 f"%{hedef*100:.0f}'in "
                 + ("ÜZERİNDE (geçmiş destekliyor)." if cagr >= hedef
                    else "ALTINDA (hızlanma varsayılıyor)."))
    return {"hedef_buyume": round(hedef, 3),
            "10y_surdurme_baz_orani": oran,
            "mesaj": mesaj, "gecmisle_kiyas": kiyas,
            "not": "Oranlar yaklaşık literatür özetidir (Mauboussin Base "
                   "Rate Book tarzı) [TÜRETİLMİŞ]; boyuta göre değişir."}


def analyze_call_text(text):
    """
    KAZANÇ ÇAĞRISI DİL ANALİZİ (hafif sözlük yöntemi): hedge kelimeleri vs
    güven kelimeleri + rehber (guidance) geçişleri. Kullanıcı metni yapıştırır
    (IR sayfası/haber); derin nitel okuma Claude'a, ilk 6000 karakter pakete.
    """
    if not text or len(text.strip()) < 200:
        return None
    low = text.lower()
    hedge = ("challenging", "headwind", "uncertain", "softness", "cautious",
             "difficult", "slowdown", "muted", "weaker", "decline", "delay",
             "pressure", "volatil")
    conf = ("record", "strong", "momentum", "raise", "accelerat", "exceed",
            "robust", "confident", "beat", "outperform", "all-time")
    h = sum(low.count(w) for w in hedge)
    c = sum(low.count(w) for w in conf)
    guide = sum(low.count(w) for w in ("guidance", "outlook", "reiterate",
                                       "raising", "lowering", "withdraw"))
    top = h + c
    ton = ("NÖTR" if top == 0 else
           "TEMKİNLİ" if h >= c * 1.5 else
           "GÜVENLİ" if c >= h * 1.5 else "NÖTR")
    return {"ton": ton, "hedge_sayisi": h, "guven_sayisi": c,
            "rehber_gecisi": guide,
            "not": "Sözlük tabanlı kaba sinyal; bağlamlı okuma Claude "
                   "raporunda.",
            "metin_alinti": text.strip()[:6000]}


def normalized_earnings(m, a):
    """
    NORMALİZE KAZANÇ MOTORU (tam sürüm): SEC 10-K serisi varsa 10 yıla kadar
    faaliyet marjı tarihi kullanılır (Yahoo yalnız ~4 yıl verir). Medyan marj
    = 'orta-döngü' kazanç gücü. Zirve/dip kazançla yapılan her çarpan
    yanıltır; normalize EPS ve normalize taban değer bu yanılsamayı kırar.
    Tüm şirketlere uygulanır — istikrarlı şirkette normalize ≈ güncel çıkar,
    zararsızdır; çevrimselde hayat kurtarır.
    """
    rev_now = latest(m.get("rev") or {})
    price = m.get("price")
    if not rev_now:
        return None
    ey = m.get("_edgar_yillik") or {}
    e_rev, e_op = ey.get("gelir") or {}, ey.get("faaliyet_kari") or {}
    ort = sorted(set(e_rev) & set(e_op))
    marjlar, kaynak = {}, None
    if len(ort) >= 5:
        marjlar = {int(y): e_op[y] / e_rev[y] for y in ort if e_rev[y]}
        kaynak = f"SEC 10-K ({len(marjlar)} yıl)"
    else:
        om = m.get("om") or {}
        if len(om) >= 3:
            marjlar = {int(y): v for y, v in om.items()}
            kaynak = f"Yahoo ({len(marjlar)} yıl — kısa seri, temkinli oku)"
    if len(marjlar) < 3:
        return None
    med = _medyan(marjlar.values())          # BULGU U: gerçek medyan
    cur = marjlar[max(marjlar)]
    if med <= 0:
        return {"uygulanamaz": "Medyan faaliyet marjı negatif — normalize "
                               "kazanç anlamsız (yapısal zarar)."}
    sapma = cur / med - 1
    durum = ("ZİRVE ÜSTÜ" if sapma > 0.25 else
             "DİP ALTI" if sapma < -0.25 else "NORMAL BANT")
    eff = m.get("eff_tax", 0.21)
    n_nopat = rev_now * med * (1 - eff)
    # Y1: fiyat KALDIRAÇLI — payda faiz-sonrası olmalı; aksi hâlde
    # normalize F/K borçlu şirkette yapısal düşük çıkar.
    _fz = m.get("int_exp")
    n_net = n_nopat - (abs(_fz) * (1 - eff) if _fz else 0.0)
    dil = m.get("dil_shares_seri") or {}
    sh = dil.get(max(dil)) if dil else m.get("shares")
    n_eps = n_net / sh if sh else None
    n_pe = (price / n_eps) if (n_eps and n_eps > 0 and price) else None
    n_taban = None
    if sh and a.get("discount"):
        n_taban = (n_nopat / a["discount"] - (m.get("net_debt") or 0)) / sh
    return {"kaynak": kaynak,
            "marj_seri": {str(y): round(v, 4)
                          for y, v in sorted(marjlar.items())},
            "medyan_marj": round(med, 4), "guncel_marj": round(cur, 4),
            "sapma": round(sapma, 3), "durum": durum,
            "normalize_eps": round(n_eps, 2) if n_eps else None,
            "normalize_fk": round(n_pe, 1) if n_pe else None,
            "raporlanan_fk": m.get("pe_t"),
            "normalize_baz": ("vergi-sonrası faiz düşüldü — raporlanan F/K ile kıyaslanabilir" if m.get("int_exp") else "KALDIRAÇSIZ (faiz verisi yok)"),
            "normalize_taban_deger": (round(n_taban, 2)
                                      if n_taban and n_taban > 0 else None),
            "yorum": ("Güncel marj medyanın %{:+.0f} {} — {} durumundayken "
                      "raporlanan kârla yapılan çarpanlar {}."
                      .format(abs(sapma) * 100,
                              "üstünde" if sapma >= 0 else "altında",
                              durum,
                              "iyimser yanılır (zirve kazancı kalıcı sanılır)"
                              if durum == "ZİRVE ÜSTÜ" else
                              "kötümser yanılır (dip kazanç kalıcı sanılır)"
                              if durum == "DİP ALTI" else
                              "makul temsil eder"))}


def forward_bands(m, b):
    """
    İLERİYE BAKIŞ + TARİHSEL F/K BANDI: (1) Analist forward EPS dağılımı
    (düşük/ortalama/yüksek) — dağılım genişse belirsizlik yüksek, tek forward
    F/K'ya yaslanma. (2) Tarihsel F/K bandı: yıllık ort. fiyat / yıllık EPS
    serisi — bugünkü F/K bandın neresinde? Bandın üstü = çarpan genişlemesi
    (temel değil, duygu). SEC serisi varsa EPS derinliği artar.
    """
    out = {}
    price = m.get("price")
    # 1) forward EPS dağılımı
    ee = b.get("earn_est")
    try:
        if ee is not None and not getattr(ee, "empty", True):
            idx = [str(i) for i in ee.index]
            cols = {str(c).lower(): c for c in ee.columns}
            for key, ad in (("0y", "bu_yil"), ("+1y", "gelecek_yil")):
                if key in idx:
                    r = ee.iloc[idx.index(key)]
                    avg = _num(r.get(cols.get("avg")))
                    lo = _num(r.get(cols.get("low")))
                    hi = _num(r.get(cols.get("high")))
                    n_ = _num(r.get(cols.get("numberofanalysts")))
                    d = {"ort": avg, "dusuk": lo, "yuksek": hi,
                         "analist": int(n_) if n_ else None}
                    if avg and lo is not None and hi is not None and avg != 0:
                        d["genislik"] = round((hi - lo) / abs(avg), 3)
                    out[ad] = d
            g1 = (out.get("gelecek_yil") or {}).get("ort")
            g0 = (out.get("bu_yil") or {}).get("ort")
            f_eps = g1 or g0
            if f_eps and price and f_eps > 0:
                out["forward_fk"] = round(price / f_eps, 1)
            gen = (out.get("gelecek_yil") or {}).get("genislik")
            if gen is not None:
                out["belirsizlik"] = ("YÜKSEK — analistler ayrışıyor"
                                      if gen > 0.35 else
                                      "orta" if gen > 0.15 else "düşük")
    except Exception:
        pass

    # 2) tarihsel F/K bandı (yıllık ort. fiyat / yıllık EPS)
    try:
        hist = b.get("hist")
        ey = m.get("_edgar_yillik") or {}
        ni_y = ey.get("net_kar") or (m.get("ni") or {})
        _sh_edgar = ey.get("seyreltilmis_hisse")
        sh_y = _sh_edgar or (m.get("dil_shares_seri") or {})
        _sh_kaynak = "edgar" if _sh_edgar else "yahoo"
        if hist is not None and len(hist) and ni_y and sh_y:
            close = hist["Close"].dropna()
            seri = {}
            for y in sorted(set(ni_y) & set(sh_y)):
                if not ni_y.get(y) or not sh_y.get(y) or ni_y[y] <= 0:
                    continue
                sel = close[close.index.year == int(y)]
                if len(sel) < 30:
                    continue
                # ── B3-7 BAĞLANDI ──────────────────────────────
                # Hisse adedi o yılın RAPORLANMIŞ adedi; fiyat serisi
                # ise bölünme-düzeltmeli. Aynı tabana getirilmezse
                # F/K bölünme oranı kadar yanlış çıkar.
                # O6: çarpan yalnız AS-FILED (EDGAR) adede; Yahoo serisinin bölünme
                # durumu belirsiz → çifte düzeltme riskine karşı 1.0 (not aşağıda).
                _bc = (bolunme_carpani(m.get("splits"), int(y))
                       if _sh_kaynak == "edgar" else 1.0)
                eps_y = ni_y[y] / (sh_y[y] * _bc)
                if eps_y > 0:
                    seri[int(y)] = round(float(sel.mean()) / eps_y, 1)
            if len(seri) >= 3:
                v = sorted(seri.values())
                lo_, md_, hi_ = v[0], _medyan(v), v[-1]   # BULGU U
                band = {"seri": seri, "dusuk": lo_, "medyan": md_,
                        "yuksek": hi_}
                if _sh_kaynak == "yahoo":
                    band["hisse_kaynak_notu"] = (
                        "Yahoo hisse serisi — bölünme düzeltmesi DOĞRULANAMADI, "
                        "çarpan uygulanmadı; bölünmeli yıllarda bandı temkinli oku (O6).")
                pe_now = m.get("pe_t")
                if pe_now and hi_ > lo_:
                    band["konum_0dip_1zirve"] = round(
                        (pe_now - lo_) / (hi_ - lo_), 2)
                out["fk_tarihsel_bant"] = band
    except Exception:
        pass
    return out or None


# ── Kayıt defteri (kalibrasyon) ──────────────────────────────────────────
GUNLUK_DOSYA = "analiz_gunlugu.json"


def gunluk_oku():
    """Yerel analiz günlüğünü oku; bozuksa/yoksa boş liste (asla çökmez)."""
    import os
    try:
        if not os.path.exists(GUNLUK_DOSYA):
            return []
        with open(GUNLUK_DOSYA, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def gunluk_yaz(kayit):
    """Analiz kaydını günlüğe ekle (son 200 tutulur). Hata → sessiz geç."""
    try:
        data = gunluk_oku()
        data.append(kayit)
        with open(GUNLUK_DOSYA, "w", encoding="utf-8") as f:
            json.dump(data[-200:], f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def kayit_coz(kayitlar, hist_getir=None):
    """
    KARAR KAYITLARINI SONUÇLA EŞLEŞTİR (kalibrasyon döngüsünün 1. yarısı).
    Ufku dolmuş, henüz çözülmemiş 'karar' kayıtları için: kayıt tarihinden
    ufuk sonuna kadar fiyat, kaydedilen DEĞER ÇIPASINA ulaştı mı?
    Ulaştıysa sonuc=1, ulaşmadıysa 0. Fiyat serisi yoksa çözülmez —
    tahmin edilmez.
    """
    bugun = pd.Timestamp.now().normalize()
    cozulen = 0
    for k in kayitlar:
        if k.get("tip") != "karar" or k.get("cozuldu"):
            continue
        try:
            bit = pd.Timestamp(k.get("ufuk_bitis"))
            bas = pd.Timestamp(k.get("tarih")[:10])
            cipa = _num(k.get("deger_cipa"))
        except Exception:
            continue
        if cipa is None or bit > bugun:
            continue
        h = hist_getir(k.get("ticker")) if hist_getir else None
        if h is None or "Close" not in getattr(h, "columns", []):
            continue
        try:
            idx = h.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            s = pd.Series(h["Close"].values, index=idx).dropna()
            pencere = s[(s.index >= bas) & (s.index <= bit)]
            if pencere.empty:
                continue
            # ── B8-2 ─────────────────────────────────────────────────
            # Eski kod başarıyı `pencere.max() >= cipa` ile belirliyordu —
            # yani DOKUNMA kuralı. Kullanıcının zihninde tarttığı olay ise
            # "vade sonunda orada olacak mı". Yansıma ilkesi gereği
            # sürüklenmesiz halde bu TAM 2 KAT fark eder (ölçüm: +%10
            # hedefte bitiş %26.3 vs dokunma %52.5). Kalibrasyon eğrisi
            # yapay "az güvenli" çıkar → geri besleme kalibrasyonu BOZAR.
            k["sonuc_dokunma"] = int(float(pencere.max()) >= cipa)
            k["sonuc_bitis"] = int(float(pencere.iloc[-1]) >= cipa)
            k["sonuc"] = (k["sonuc_dokunma"]
                          if k.get("cozum_kurali") == "dokunma"
                          else k["sonuc_bitis"])
            k["sonuc_zirve"] = round(float(pencere.max()), 2)
            k["sonuc_kapanis"] = round(float(pencere.iloc[-1]), 2)
            k["cozuldu"] = True
            cozulen += 1
        except Exception:
            continue
    return cozulen


def kalibrasyon(kayitlar):
    """
    BRIER SKORU + BANT KALİBRASYONU (döngünün 2. yarısı).
    Brier = ortalama (olasılık − sonuç)². 0 mükemmel, 0.25 = yazı-tura.
    Bant kalibrasyonu: '%70 dediklerimin kaçı gerçekleşti?' — asıl
    öğretici olan budur. Az örnekle anlamsızdır; eşik açıkça yazılır.
    """
    coz = [k for k in kayitlar
           if k.get("tip") == "karar" and k.get("cozuldu")
           and _num(k.get("olasilik")) is not None
           and k.get("sonuc") in (0, 1)]
    if not coz:
        return None
    n = len(coz)
    brier = sum((float(k["olasilik"]) / 100 - k["sonuc"]) ** 2
                for k in coz) / n
    bantlar = {}
    for k in coz:
        p = float(k["olasilik"]) / 100
        b = min(int(p * 10) * 10, 90)
        bantlar.setdefault(b, []).append(k["sonuc"])
    bant_out = [{"bant": f"%{b}-{b+10}", "adet": len(v),
                 "gerceklesen": round(100 * sum(v) / len(v), 0),
                 "beklenen": b + 5}
                for b, v in sorted(bantlar.items())]
    # ── B8-5 ─────────────────────────────────────────────────────────
    # 0.25 YALNIZCA taban oran %50 iken doğru referanstır. Taban %30 ise
    # her zaman "%30" diyen bir tahminci 0.30×0.70 = 0.21 alır — yani
    # 0.25'i geçmek TRİVİALDİR ve yanıltıcı bir başarı hissi verir.
    # Doğru ölçüt: BS_clim = p̄(1−p̄) ve BECERİ SKORU = 1 − BS/BS_clim.
    _taban = sum(k["sonuc"] for k in coz) / n
    _bs_clim = _taban * (1 - _taban)
    _bss = (1 - brier / _bs_clim) if _bs_clim > 1e-9 else None
    return {"adet": n, "brier": round(brier, 4),
            "taban_oran": round(_taban, 3),
            "brier_taban": round(_bs_clim, 4),
            "beceri_skoru": (round(_bss, 3) if _bss is not None else None),
            "referans": {"mukemmel": 0.0,
                         "taban_oran_tahmincisi": round(_bs_clim, 4)},
            "bantlar": bant_out,
            # B8-6: 15 tahmin / 10 bant = bant başına 1.5 gözlem; n=15'te
            # tek bantta bile hata payı ±25 puandır. Eşik 50'ye çıkarıldı.
            "yeterli_mi": n >= 50,
            "not": ("Brier 0.25 = yazı-tura; altı iyidir. Bant tablosu asıl "
                    "öğreticidir: '%70 dediklerinin ~%70'i gerçekleşiyor "
                    "mu?'. En az ~15 çözülmüş karar birikmeden bu sayılar "
                    "ANLAMLI DEĞİLDİR — az örnekte gürültü baskındır."
                    if n < 15 else
                    "Bant tablosunu oku: beklenen ile gerçekleşen sürekli "
                    "ayrışıyorsa aşırı/eksik güven var demektir.")}


def mutabakat_raporu(m, b, tolerans=0.02):
    """
    MUTABAKAT (araştırma B5): Yahoo türevli değerler ile SEC EDGAR'ın
    RAPORLADIĞI değerler kalem kalem karşılaştırılır. Sapma toleransı
    aşarsa kırmızı satır düşer — 'sessiz yanlış veri' böyle yakalanır.
    Tolerans varsayılan ±%2 (yuvarlama/sunum farkı payı).
    """
    ey = ((b.get("edgar") or {}).get("yillik") or {})
    if not ey:
        return None
    esles = [("Gelir", "rev", "gelir"), ("Net kâr", "ni", "net_kar"),
             ("Faaliyet kârı", "oi", "faaliyet_kari"),
             ("CFO", "cfo", "cfo"),
             # FIX: "eq"/"dsh" anahtarları m sözlüğünde HİÇ YOK — bu iki
             # kalem sessizce atlanıyordu (6 kalemlik mutabakat fiilen
             # 4 kalem çalışıyordu). Doğru anahtarlar bağlandı.
             ("Özsermaye", "eq_seri", "ozsermaye"),
             ("Seyreltilmiş hisse", "dil_shares_seri",
              "seyreltilmis_hisse")]
    satir = []
    for ad, mkey, ekey in esles:
        yseri = m.get(mkey) or {}
        eseri = ey.get(ekey) or {}
        if not isinstance(yseri, dict) or not eseri:
            continue
        ortak = sorted(set(yseri) & set(eseri))
        if not ortak:
            continue
        yil = ortak[-1]
        yv, ev = _num(yseri.get(yil)), _num(eseri.get(yil))
        if yv is None or ev is None or ev == 0:
            continue
        sapma = abs(yv - ev) / abs(ev)
        satir.append({"kalem": ad, "yil": yil, "hesaplanan": yv,
                      "sec_raporlanan": ev, "sapma": round(sapma, 4),
                      "durum": "OK" if sapma <= tolerans else "SAPMA"})
    if not satir:
        return None
    kotu = [s for s in satir if s["durum"] == "SAPMA"]
    return {"satirlar": satir, "sapan": len(kotu), "tolerans": tolerans,
            "not": (f"±%{tolerans*100:.0f} toleransla SEC'in raporladığı "
                    f"değerlere karşı mutabakat. "
                    + ("Tüm kalemler tutuyor." if not kotu else
                       f"{len(kotu)} kalemde sapma var — bu kalemlere "
                       f"dayanan oranları temkinli oku ve 10-K'dan "
                       f"doğrula."))}


def _stooq_fiyat(ticker):
    """Yahoo bloklanırsa fiyat için ücretsiz/anahtarsız yedek (Stooq).
    Yalnız kapanış serisi verir; başarısızsa None (sessiz bozulma yok)."""
    try:
        import requests
        from io import StringIO
        url = (f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d")
        r = requests.get(url, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text or r.text[:1] == "<":
            return None                      # HTML hata sayfası/blok
        df = pd.read_csv(StringIO(r.text))
        if df is None or df.empty or "Close" not in df:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date")[["Close"]].dropna()
    except Exception:
        return None


DURUM_DOSYA = "izleme_durumu.json"


def durum_oku():
    import os
    try:
        if not os.path.exists(DURUM_DOSYA):
            return {}
        with open(DURUM_DOSYA, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def durum_yaz(ticker, anlik):
    try:
        d = durum_oku()
        d[ticker] = anlik
        with open(DURUM_DOSYA, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def anlik_durum(m):
    """Bu ziyaretin 'fotoğrafı' — bir sonraki açılışta farkı çıkarmak için."""
    kd = (m.get("kademe") or {})
    d1 = ((kd.get("dilimler") or [{}])[0]).get("seviye") or kd.get(
        "izleme_esigi")
    st_ = (m.get("stop") or {})
    o8 = (m.get("olaylar_8k") or {}).get("olaylar") or []
    kz = None
    for e in ((m.get("catalysts") or {}).get("gelecek") or []):
        if "Kazanç" in e.get("olay", ""):
            kz = e.get("tarih"); break
    return {"tarih": str(pd.Timestamp.now().date()),
            "fiyat": m.get("price"), "hukum": (m.get("verdict") or [None])[0],
            "skor": m.get("score"), "kalite": m.get("quality"),
            "dilim1": d1,
            "trailing": (st_.get("trailing") or {}).get("seviye"),
            "giris_stop": (st_.get("giris_stop") or {}).get("seviye"),
            "son_8k": (o8[0].get("tarih") if o8 else None),
            "kazanc_tarihi": kz}


def durum_farki(m):
    """
    SON ZİYARETTEN BERİ NE DEĞİŞTİ (poll tabanlı, lokal, ücretsiz):
    fiyat bir eşiği KESTİ mi, yeni 8-K düştü mü, hüküm değişti mi, kazanç
    tarihi geçti mi. Push altyapısı yok — açılışta hesaplanır.
    """
    eski = durum_oku().get(m.get("ticker") or "")
    if not eski:
        return None
    yeni, degis = anlik_durum(m), []
    f0, f1 = eski.get("fiyat"), yeni.get("fiyat")
    if f0 and f1:
        d = f1 / f0 - 1
        if abs(d) >= 0.03:
            degis.append(("bilgi", f"Fiyat son bakıştan beri %{d*100:+.1f} "
                                   f"({eski['tarih']}: {f0:,.2f} → "
                                   f"{f1:,.2f})"))
        for ad, sev in (("1. alım dilimi", eski.get("dilim1")),
                        ("giriş stopu", eski.get("giris_stop")),
                        ("trailing stop", eski.get("trailing"))):
            if sev and f0 and f1 and (f0 - sev) * (f1 - sev) < 0:
                yon = "AŞAĞI kesti" if f1 < sev else "YUKARI kesti"
                tur = "uyarı" if "stop" in ad else "bilgi"
                degis.append((tur, f"Fiyat {ad} seviyesini ({sev:,.2f}) "
                                   f"{yon}."))
    if eski.get("son_8k") != yeni.get("son_8k") and yeni.get("son_8k"):
        degis.append(("uyarı", f"Yeni SEC 8-K bildirimi var "
                               f"({yeni['son_8k']}) — Katalizör sekmesinde "
                               f"Item'ına bak."))
    if eski.get("hukum") and eski["hukum"] != yeni.get("hukum"):
        degis.append(("uyarı", f"HÜKÜM değişti: {eski['hukum']} → "
                               f"{yeni['hukum']}"))
    ek = eski.get("kazanc_tarihi")
    if ek:
        try:
            if pd.Timestamp(ek) < pd.Timestamp.now():
                degis.append(("bilgi", f"Beklenen kazanç tarihi ({ek}) "
                                       f"GEÇTİ — sonuçlar açıklandı mı?"))
        except Exception:
            pass
    return {"onceki_tarih": eski.get("tarih"),
            "degisimler": degis} if degis else None


def monte_carlo_dcf(fcf0, a, net_debt, shares, nd_adjust, price,
                    n=10000, seed=42):
    """
    Adil değeri TEK sayı değil OLASILIK DAĞILIMI olarak üret: büyüme,
    iskonto ve terminal varsayımları belirsizlik bantlarıyla n kez
    örneklenir. Dar dağılım = sağlam değerleme; geniş = varsayıma bağımlı.
    """
    if not fcf0 or fcf0 <= 0 or not shares:
        return None
    # ── B4-7: KORELASYONLU ÖRNEKLEME ─────────────────────────────────
    # Değer büyümeyle ARTAR (∂V/∂g>0), iskontoyla AZALIR (∂V/∂d<0) —
    # işaretler ters olduğu için POZİTİF korelasyon bandı DARALTIR (yüksek
    # büyüme yüksek betayla gelir, birbirini dengeler). ÖLÇÜM: bağımsız
    # bant genişliği 129.2 · ρ=+0.4'te 104.0 → bağımsız örnekleme bandı
    # ~%24 ABARTIYORDU.
    # ── B4-8: yol sayısı 2000 → 10000 (P95 tohuma göre ±3.67 oynuyordu)
    # ── B4-9: terminal kısıtı SANSÜR değil reddet-ve-yeniden-çek
    import numpy as np
    rng = np.random.default_rng(seed)
    _rho = 0.30
    _z = rng.multivariate_normal([0.0, 0.0], [[1.0, _rho], [_rho, 1.0]], n)
    g = np.clip(a["growth"] + max(0.03, abs(a["growth"]) * 0.35) * _z[:, 0],
                -0.10, 0.40)
    d = np.clip(a["discount"] + 0.012 * _z[:, 1], 0.06, 0.18)
    t = rng.normal(a["terminal"], 0.004, n)
    for _ in range(6):
        _kotu = (t < 0.005) | (t > d - 0.015)
        if not _kotu.any():
            break
        t[_kotu] = rng.normal(a["terminal"], 0.004, int(_kotu.sum()))
    t = np.minimum(np.clip(t, 0.005, None), d - 0.015)
    yrs = int(a["years"])
    # ARAŞTIRMA TURU A2: B4-2 kısıtı MC'de de aktif — nokta-DCF ile aynı
    # model (aksi hâlde MC medyanı sistematik yukarı kayardı, Q bulgusu).
    _rt = _num((a or {}).get("roic_terminal"))
    # ── DENETİM BULGUSU Q (YÜKSEK): MODEL TUTARSIZLIĞI ────────────────
    # dcf_fair_value KADEMELİ FADE uyguluyor (ilk yarı sabit büyüme, ikinci
    # yarı terminale doğrusal iniş) ama Monte Carlo BÜTÜN yıllarda SABİT
    # büyüme kullanıyordu. Sonuç: MC dağılımı nokta tahminden sistematik
    # olarak YÜKSEK çıkıyordu (ölçüm: medyan %14.4 yukarıda). İki motor aynı
    # şirket için farklı model çalıştırıyordu. MC bandı başlık değer metriği
    # yapıldığı için bu sapma doğrudan kullanıcının gördüğü sayıyı bozuyordu.
    # DÜZELTME: fade mantığı dcf_fair_value ile BİREBİR aynı (vektörel).
    hizli = max(1, yrs // 2)
    fcf = np.full(n, float(fcf0))
    pv = np.zeros(n)
    for i in range(1, yrs + 1):
        if i <= hizli:
            g_i = g
        else:                                   # kademeli iniş (fade)
            pay = (i - hizli) / max(1, yrs - hizli)
            g_i = g + (t - g) * pay
        fcf = fcf * (1 + g_i)
        pv += fcf / (1 + d) ** (i - 0.5)      # B4-1 ile AYNI model
    _tvf = 1.0   # O3: nokta-DCF paritesi — RONIC çarpanı FCF tabanında çifte sayımdı
    ev = pv + fcf * (1 + t) * _tvf / (d - t) / (1 + d) ** (yrs - 0.5)
    if nd_adjust and net_debt:
        ev = ev - net_debt
    fair = ev / shares
    fair = fair[np.isfinite(fair)]
    if fair.size < 100:
        return None
    q = np.percentile(fair, [5, 25, 50, 75, 95])
    return {"p5": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
            "p95": round(float(q[4]), 2),
            "yukari_potansiyel_olasiligi": (round(float((fair > price).mean()), 3)
                                            if price else None),
            "orneklem": int(fair.size),
            "dagilim": [round(float(x), 2)
                        for x in np.percentile(fair, np.arange(2, 100, 2))],
            "not": ("10.000 senaryo; büyüme/iskonto/terminal belirsizlik "
                    "bandıyla örneklendi. Fade + terminal reinvestman "
                    "kısıtı nokta-DCF ile AYNIDIR — medyan, nokta "
                    "tahminin etrafında oturur.")}


def _fair_makul_mu(fair, price):
    """DCF adil değeri fiyata göre MAKUL bantta mı? Absürt değerler (fiyatın
    %25'inin altı veya 3 katının üstü) izleme eşiği/dilim çıpası OLARAK
    KULLANILMAZ — bu, motorun DCF varsayım-duyarlılığından doğan saçma
    sayıların plana sızmasını önler (ör. $233 hisseye $4.82 eşik)."""
    if not fair or not price or price <= 0:
        return False
    return 0.25 * price <= fair <= 3.0 * price


def atr_serisi(hist, n=22):
    """Wilder ATR (1978 orijinal yumuşatma: ewm alpha=1/n). Rapor önerisi:
    orta vade (3-4 ay) için n=22 ≈ 1 işlem ayı. High/Low yoksa kapanış
    farkıyla YAKLAŞIK hesaplanır (etiketi stop planında düşülür)."""
    try:
        if hist is None or len(hist) < n + 5 or "Close" not in hist:
            return None
        c = hist["Close"]
        h = hist["High"] if "High" in hist else c
        l = hist["Low"] if "Low" in hist else c
        # ── B7-4 ──────────────────────────────────────────────────────
        # ewm(alpha=1/n, adjust=False) Wilder özyinelemesiyle CEBİRSEL
        # OLARAK AYNIDIR — ama pandas seriyi İLK GÖZLEMLE tohumlar, Wilder
        # ise ilk n TR'nin ARİTMETİK ORTALAMASIYLA. min_periods=n bunu
        # GİDERMEZ, yalnız ilk n−1 barı gizler. ÖLÇÜM: bar 21'de %+18.4
        # sapma; %1'in altına ancak bar 84'te iniyor.
        # ── B7-3 ──────────────────────────────────────────────────────
        # High/Low yoksa TR kapanış farkıyla yaklaşılıyor; bu bar-içi
        # genişliği TAMAMEN atar. ÖLÇÜM: ATR %51.7 DÜŞÜK → 3×ATR stop
        # mesafesi %6.7 yerine %3.2 ve pozisyon %107 DAHA BÜYÜK. İki hata
        # da AYNI YÖNDE. Sezgisel 1.6× telafi uygulanır [TÜRETİLMİŞ].
        _tam = ("High" in hist and "Low" in hist)
        pc = c.shift(1)
        tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()],
                       axis=1).max(axis=1).dropna()
        if len(tr) < n:
            return None
        _deger = [float("nan")] * len(tr)
        _onc = float(tr.iloc[:n].mean())          # WILDER TOHUMU
        _deger[n - 1] = _onc
        for _i in range(n, len(tr)):
            _onc = (_onc * (n - 1) + float(tr.iloc[_i])) / n
            _deger[_i] = _onc
        atr = pd.Series(_deger, index=tr.index).dropna()
        if not len(atr):
            return None
        if not _tam:
            atr = atr * 1.6
        return atr
    except Exception:
        return None


def destek_direnc(hist, price, atr, k=5, min_guc=55, min_dokunma=2,
                  maks_bolge=3):
    """
    KANITA DAYALI DESTEK/DİRENÇ HARİTASI (araştırma karar listesi):
      • Swing dip/tepe (fractal, k=5 → 11 barlık pencere) → 0.5×ATR
        toleransla BÖLGELERE kümelenir [trading-range kanıtı: Brock-
        Lakonishok-LeBaron 1992 — güçlü].
      • Güç skoru [TÜRETİLMİŞ sezgisel 0-100]: dokunma sayısı + dokunma
        hacmi + tazelik + polarity (hem dip hem tepe olarak test edilmiş
        seviye — kırıl/geri-test yaklaşığı).
      • Çakışma etiketleri: 200G/50G ortalama, 52 hafta dip/zirve
        [George-Hwang 2004], yuvarlak sayı (STOP KÜMELENMESİ riski —
        Osler; Bhattacharya-Holden-Jacobsen 2012).
    Fibonacci ve pivot noktaları BİLİNÇLİ olarak yok (kanıt: folklor).
    Seviyeler ÇİZGİ değil BÖLGEdir; yön garantisi değildir.
    """
    try:
        if hist is None or not price or len(hist) < 60:
            return None
        c = hist["Close"].dropna()
        h = (hist["High"] if "High" in hist else hist["Close"]).dropna()
        l = (hist["Low"] if "Low" in hist else hist["Close"]).dropna()
        vol = hist["Volume"] if "Volume" in hist else None
        son = c.index[-1]
        if getattr(son, "tzinfo", None) is not None:
            son = son.tz_localize(None)
        atr = _num(atr) or float(c.pct_change().dropna().std()
                                 * c.iloc[-1]) or price * 0.02
        tol = 0.5 * atr

        w = 2 * k + 1
        lo_min = l.rolling(w, center=True).min()
        hi_max = h.rolling(w, center=True).max()
        noktalar = []                       # (seviye, tarih, tur, hacim)
        for i in range(k, len(c) - k):
            try:
                ts = l.index[i]
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_localize(None)
                v_ = (float(vol.iloc[i]) if vol is not None
                      and pd.notna(vol.iloc[i]) else None)
                # 5. eleman = bar indeksi: ayrı ZİYARET sayımı için gerekli
                # (bkz. aşağıdaki _ziyaret_say — bitişik pivotlar tek ziyaret)
                if pd.notna(lo_min.iloc[i]) and l.iloc[i] <= lo_min.iloc[i]:
                    noktalar.append((float(l.iloc[i]), ts, "dip", v_, i))
                if pd.notna(hi_max.iloc[i]) and h.iloc[i] >= hi_max.iloc[i]:
                    noktalar.append((float(h.iloc[i]), ts, "tepe", v_, i))
            except Exception:
                continue
        if not noktalar:
            return None
        noktalar.sort(key=lambda x: x[0])

        kumeler, aktif = [], [noktalar[0]]
        for p in noktalar[1:]:
            merkez = sum(x[0] for x in aktif) / len(aktif)
            if abs(p[0] - merkez) <= tol:
                aktif.append(p)
            else:
                kumeler.append(aktif)
                aktif = [p]
        kumeler.append(aktif)

        vol_ort = (float(vol.dropna().mean())
                   if vol is not None and len(vol.dropna()) else None)
        ma50 = (float(c.rolling(50).mean().iloc[-1])
                if len(c) >= 50 else None)
        ma200 = (float(c.rolling(200).mean().iloc[-1])
                 if len(c) >= 200 else None)
        lo52 = float(l.iloc[-252:].min()) if len(l) >= 60 else None
        hi52 = float(h.iloc[-252:].max()) if len(h) >= 60 else None
        adim = (25.0 if price >= 500 else 10.0 if price >= 200
                else 5.0 if price >= 50 else 1.0 if price >= 10 else 0.5)

        bolgeler = []
        for kum in kumeler:
            svler = [x[0] for x in kum]
            b_alt, b_ust = min(svler), max(svler)
            if b_ust - b_alt < tol * 0.4:            # tekil → dar bant aç
                b_alt, b_ust = b_alt - tol * 0.3, b_ust + tol * 0.3
            # ── SINIR DURUMU DÜZELTMESİ ──────────────────────────────
            # ESKİ: dokunma = len(kum) → kümedeki her PİVOT bir dokunma
            # sayılıyordu. Ama fractal koşulu (l <= 11-barlık rolling min)
            # düz/bitişik dipte ARDIŞIK BARLARIN HEPSİNİ pivot işaretler:
            # yer-gerçeği testinde TEK ziyaret 4 dokunma çıktı ve
            # 'dokunma ≥ 2' filtresini tek başına geçti.
            # YENİ: birbirine k bardan yakın pivotlar AYNI ziyarettir.
            _idx = sorted(x[4] for x in kum if len(x) > 4)
            if _idx:
                dokunma, _son_i = 1, _idx[0]
                for _i2 in _idx[1:]:
                    if _i2 - _son_i > k:          # yeni ziyaret
                        dokunma += 1
                        _son_i = _i2
            else:
                dokunma = len(kum)                # geriye dönük güvenlik
            son_ts = max(x[1] for x in kum)
            gun = max(0, int((son - son_ts).days))
            hacimler = [x[3] for x in kum if x[3]]
            h_orani = (round(sum(hacimler) / len(hacimler) / vol_ort, 2)
                       if hacimler and vol_ort else None)
            polarity = (len({x[2] for x in kum}) == 2)

            etiket = []
            guc = 20.0 + 12 * min(dokunma, 5)
            if h_orani is not None:
                guc += 10 if h_orani > 1.2 else (5 if h_orani > 1.0 else 0)
            guc += (15 if gun <= 60 else 10 if gun <= 180
                    else 5 if gun <= 365 else 0)
            if polarity:
                guc += 10
                etiket.append("polarity (iki yönlü test)")
            for ad_ma, v_ma in (("MA50", ma50), ("MA200", ma200)):
                if v_ma and b_alt - tol <= v_ma <= b_ust + tol:
                    guc += 8
                    etiket.append(f"{ad_ma} çakışması")
            for ad52, v52 in (("52h dip", lo52), ("52h zirve", hi52)):
                if v52 and b_alt - tol <= v52 <= b_ust + tol:
                    guc += 8
                    etiket.append(ad52)
            r0 = math.ceil((b_alt - 1e-9) / adim) * adim
            if r0 <= b_ust + 1e-9:
                etiket.append(f"yuvarlak {r0:g} — stop kümelenmesi riski")
            bolgeler.append({"bant_alt": round(b_alt, 2),
                             "bant_ust": round(b_ust, 2),
                             # guc: 0-100'e KIRPILMIŞ (filtre eşikleri buna
                             # göre kalibre; davranış değişmesin diye korundu)
                             "guc": round(min(100.0, guc), 0),
                             # guc_ham: KIRPILMAMIŞ ham toplam (teorik tavan
                             # 137). Kırpma yüzünden bölgelerin çoğu 100'de
                             # eşitleniyor ve sıralama rastgeleleşiyordu —
                             # canlı INTC verisinde üç desteğin üçü de 100'dü.
                             # Sıralama ve eşitlik bozma artık bunu kullanır.
                             "guc_ham": round(guc, 1),
                             "dokunma": dokunma,
                             "son_test_gun": gun,
                             "hacim_orani": h_orani,
                             "etiketler": etiket or None})

        # Eskiden yalnız en yakın 4 destek / 3 direnç tutuluyordu; uzaktaki
        # ama GÜÇLÜ bölgeler (ör. fiyat 150 iken 118'deki taban) listeden
        # düşüyordu. Artık 8/6 tutuluyor VE güç skoru yüksek olanlar
        # mesafeye bakılmaksızın korunuyor.
        # KALİTE FİLTRESİ (az ama öz): bölge seçimi YAKINLIĞA değil GÜCE
        # göre yapılır. Tek dokunuşluk yerler bölge sayılmaz; zayıf skorlu
        # olanlar elenir. Böylece ekranda 8-10 belirsiz çizgi yerine 2-3
        # gerçekten anlamlı bölge kalır — uzakta olsa bile güçlüyse durur.
        def _sec(liste, ters):
            iyi = [z for z in liste
                   if z["dokunma"] >= min_dokunma and z["guc"] >= min_guc]
            # Sıralama KIRPILMAMIŞ skora göre: kırpılmış 'guc' ile bölgeler
            # 100'de eşitlenip sıra rastgeleleşiyordu. Eşitlik bozucu olarak
            # dokunma sayısı ve tazelik (küçük gün = taze) kullanılır.
            _sira = lambda z: (-z.get("guc_ham", z["guc"]), -z["dokunma"],
                               z.get("son_test_gun") or 9999)
            if not iyi:                       # hiçbiri geçemediyse eşiği
                # TUR2 bulgusu: yedek liste GÜÇ sırasından ÖNCE kesiliyordu
                # → en güçlüler değil fiyat sırasındaki ilk 3 kalıyordu.
                iyi = sorted([z for z in liste if z["dokunma"] >= 2],
                             key=_sira)[:maks_bolge]
            iyi = sorted(iyi, key=_sira)[:maks_bolge]
            return sorted(iyi, key=lambda z: z["bant_ust"], reverse=ters)

        destekler = _sec([z for z in bolgeler
                          if z["bant_ust"] < price * 1.005], True)
        direncler = _sec([z for z in bolgeler
                          if z["bant_alt"] > price * 0.995], False)
        if not destekler and not direncler:
            return None
        return {"destekler": destekler or None,
                "direncler": direncler or None,
                "atr": round(atr, 2), "tolerans": round(tol, 2),
                "yontem": (f"fractal k={k}, kümeleme 0.5×ATR · "
                           f"filtre: güç≥{min_guc}, dokunma≥{min_dokunma}, "
                           f"en fazla {maks_bolge} bölge/yön"),
                "kanit_sinirlari": (
                    "KANIT DURUMU: Destek/direnç için en güçlü akademik "
                    "kanıt DÖVİZ ve GÜN-İÇİ veridedir (Osler 2000/2003 — "
                    "mekanizma emir kümelenmesidir). HİSSE + 3-4 aylık ufuk "
                    "için doğrudan kanıt YOKTUR. 'Dokunma sayısı arttıkça "
                    "seviye güçlenir' iddiasının hakemli dayanağı "
                    "bulunamadı — bu bir pratisyen geleneğidir. Ayrıca "
                    "Sullivan-Timmermann-White (1999) veri-tarama "
                    "düzeltmesi sonrası teknik kural üstünlüğünün örnek-dışı "
                    "dönemde kaybolduğunu gösterdi. Bu seviyeleri KEHANET "
                    "değil, kararı önceden yazmaya yarayan ÇIPA say."),
                "not": ("Bölgeler olasılık eğilimidir, yön garantisi "
                        "DEĞİLDİR; güç skoru sezgiseldir [TÜRETİLMİŞ]. "
                        "Fibonacci/pivot bilinçli olarak dahil edilmedi "
                        "(kanıt: folklor).")}
    except Exception:
        return None


def stop_plani(m, a):
    """
    STOP & POZİSYON PLANI (araştırma karar listesi):
      • Trailing: Chandelier Exit (LeBeau) = 22G en yüksek − çarpan×ATR22;
        yalnız yukarı hareket eder, GÜNLÜK KAPANIŞLA değerlendirilir
        (gün içi fitiller sayılmaz — Yahoo fiyatı zaten ~15 dk gecikmeli).
      • Yeni giriş stopu (dilim-1'e bağlı): 3.0×ATR volatilite stopu ile
        'en yakın destek bandının altı − 0.5×ATR tampon' adaylarından
        GENİŞ (düşük) olanı seçilir — bariz seviyenin dibine stop koymak
        kümelenme/avcılık riskidir (Osler) ve dar stop whipsaw üretir.
      • Pozisyon boyutu = risk bütçesi ÷ stop mesafesi (Tharp). %1-2 kuralı
        SAVUNMACIDIR; optimal risk stratejiye bağlıdır (Kelly eleştirisi).
    DÜRÜSTLÜK: Stop yön tahmini değildir; Kaminski-Lo'ya göre ortalamaya-
    dönüş rejiminde beklenen getiriyi DÜŞÜREBİLİR; gap'te fiyat GARANTİSİ
    YOKTUR (kazanç günü gerçek zarar stopun altında dolabilir).
    """
    if not (a or {}).get("stop_acik", True):
        return None
    hist = m.get("_hist")
    price = m.get("price")
    atr = m.get("atr22")
    if hist is None or not price or not atr:
        return {"durum": "VERI_YOK",
                "not": "Fiyat geçmişi/ATR yetersiz — stop planı üretilmedi "
                       "(uydurma seviye verilmez)."}
    yaklasik = not ("High" in hist and "Low" in hist)
    carpan = float((a or {}).get("atr_carpan") or 3.0)
    out = {"atr": round(atr, 2), "atr_pct": round(atr / price, 4),
           "carpan": carpan,
           "atr_notu": ("ATR, High/Low olmadığından kapanış farkıyla "
                        "YAKLAŞIK hesaplandı." if yaklasik else None)}
    try:
        hh = float((hist["High"] if "High" in hist
                    else hist["Close"]).dropna().iloc[-22:].max())
        ch = hh - carpan * atr
        out["trailing"] = {
            "seviye": round(ch, 2),
            "mesafe_pct": round(1 - ch / price, 3),
            "formul": (f"22G en yüksek ({hh:,.2f}) − "
                       f"{carpan:.1f}×ATR22 ({atr:,.2f})"),
            "kural": ("Elde tutanlar için: günlük KAPANIŞ bu seviyenin "
                      "altına inerse çıkışı değerlendir; seviye her gün "
                      "yeniden hesaplanır ve YALNIZ yukarı hareket eder.")}
    except Exception:
        pass

    kd = m.get("kademe") or {}
    d1 = (((kd.get("dilimler") or [{}])[0]).get("seviye")
          or kd.get("izleme_esigi") or price)
    # DOĞRULANMIŞ ÇARPAN: 34 hisse × 15 yıl walk-forward testinde 3.0-3.5×
    # ATR, dar çarpanlardan (2.0-2.5) belirgin biçimde üstün çıktı.
    g_carp = float((a or {}).get("giris_carpan") or 3.0)
    stop_atr = d1 - g_carp * atr
    # ══ GERİYE DÖNÜK TEST BULGUSU (kritik hata düzeltildi) ══════════
    # ESKİ DAVRANIŞ: aşağıdaki döngü, dilim-1'in ALTINDAKİ HERHANGİ bir
    # destek bölgesini "en yakın" sayıyordu — bölge ne kadar uzak olursa
    # olsun. Yükselen piyasada fiyat sürekli yeni zirve yaptığı için
    # bölgeler çok geride kalır; stop o uzak bölgenin DİBİNE konuyordu.
    # 13.792 sinyallik walk-forward testinde ölçülen sonuç:
    #   • stop mesafesi medyanı girişin %63 ALTI (2026'da %82)
    #   • vakaların yalnız %17,9'unda doğrulanmış 3×ATR stopu kullanılmış
    #   • 3R hedefi girişin ~%189 üstüne çıkıyor → 63 günde %1,0 isabet
    #   • stop %4,0 tetikleniyor → fiili koruma YOK
    #   • pozisyon = risk bütçesi ÷ dev mesafe → portföyün ~%1,6'sı
    # DOĞRU MANTIK (docstring'in kendi gerekçesi): yapısal stop ancak
    # dilim-1 GERÇEKTEN bir destek bölgesine oturduğunda anlamlıdır —
    # "bölge kırılırsa neden girildiği de kalmaz". Dilim-1 yüzde dilimiyse
    # ortada yapısal bir giriş gerekçesi YOKTUR, ATR stopu yönetir.
    # Ayrıca yapısal stop için GÜVENLİK TAVANI: ATR stopunun iki katından
    # (varsayılan 6×ATR) daha geniş olamaz.
    _d1 = ((kd.get("dilimler") or [{}])[0])
    en_yakin = None
    if _d1.get("cipa") == "destek bölgesi" and _d1.get("bant_alt"):
        en_yakin = {"bant_alt": _d1["bant_alt"], "guc": _d1.get("guc")}
    stop_yapi = (en_yakin["bant_alt"] - 0.5 * atr) if en_yakin else None
    stop_yapi_ham = stop_yapi
    _tavan = d1 - 2.0 * g_carp * atr          # en fazla 2× ATR stop mesafesi
    if stop_yapi is not None and stop_yapi < _tavan:
        stop_yapi = None                       # bölge çok geniş → ATR yönetir
    adaylar = [x for x in (stop_atr, stop_yapi)
               if x is not None and x < d1]
    if not adaylar:
        adaylar = [d1 - g_carp * atr]
    sec = min(adaylar)
    _guv_tabani = False
    if sec <= 0:
        # ALIM TURU BULGUSU E1: ATR fiyata göre aşırı büyükse (ATR ≥ d1/3)
        # 3×ATR hesabı NEGATİF stop üretiyordu — anlamsız ve tehlikeli çıktı.
        # Güvenlik tabanı: stop, dilim-1'in %50'sinden aşağı olamaz; bu bir
        # strateji değişikliği değil saçma-sayı korumasıdır. Not, kullanıcıyı
        # asıl konuya yönlendirir: bu volatilitede pozisyon KÜÇÜK tutulur.
        sec = d1 * 0.5
        _guv_tabani = True
        stop_yapi = None
    kaynak = ("yapı stopu: destek bandı altı − 0.5×ATR tampon"
              if stop_yapi is not None and sec == stop_yapi
              else "GÜVENLİK TABANI (%50): ATR fiyata göre aşırı — hesap "
                   "negatif stop üretti; bu oynaklıkta stop mesafesi zaten "
                   "devasa, pozisyonu ONA GÖRE KÜÇÜLT"
              if _guv_tabani
              else f"{g_carp:.1f}×ATR volatilite stopu (doğrulanmış)")
    mesafe = d1 - sec
    out["giris_stop"] = {
        "dilim1": round(d1, 2), "seviye": round(sec, 2),
        "mesafe": round(mesafe, 2),
        "mesafe_pct": round(mesafe / d1, 3) if d1 else None,
        "kaynak": kaynak,
        "atr_stop": round(stop_atr, 2),
        "yapi_stop": (round(stop_yapi, 2)
                      if stop_yapi is not None else None),
        "yapi_stop_ham": (round(stop_yapi_ham, 2)
                          if stop_yapi_ham is not None else None),
        "yapi_stop_tavan_notu": (
            (f"Yapısal aday ({stop_yapi_ham:,.2f} = destek altı − "
             f"0.5×ATR) güvenlik tavanını (2×ATR-stop mesafesi) aştığı "
             f"için ELENDİ — ATR stopu yönetiyor. Bölge o kadar uzaksa, "
             f"kırılması giriş gerekçesini de düşürürdü.")
            if (stop_yapi is None and stop_yapi_ham is not None)
            else None),
        "destek_guc": (en_yakin["guc"] if en_yakin else None),
        "not": ("İki adaydan GENİŞ (düşük) olan seçildi: bariz desteğin "
                "hemen dibine konan stop kümelenir ve avlanır (Osler); "
                "dar stop whipsaw üretir.")}

    # SABİT R HEDEFİ — doğrulama testinde 3R, direnç-bazlı hedefi açık
    # farkla yendi (beklenti 0.201 vs 0.095). Direnç hedefi ARTIK ÖNERİLMİYOR.
    # (a4) Denetim raporu "hedef = min(3R, ilk güçlü direnç)" öneriyor.
    # UYGULANMADI — çünkü bu dosyanın KENDİ out-of-sample kanıtı (34 hisse
    # × 15 yıl walk-forward) sabit 3R'nin direnç-bazlı hedefi açık farkla
    # yendiğini gösteriyor (beklenti 0.201 vs 0.095). Kendi verini genel
    # bir öneriye feda etmek geriye gidiş olur. Uzlaşma: hedef 3R'de KALIR,
    # yoldaki güçlü direnç yalnız KISMİ KÂR ALMA bilgisi olarak eklenir.
    r_kat = float((a or {}).get("r_hedef") or 3.0)
    _hlv = d1 + r_kat * mesafe
    _yol = None                    # hedefe giden yoldaki ilk güçlü direnç
    for z in ((m.get("sr") or {}).get("direncler") or []):
        if z.get("bant_alt") and d1 < z["bant_alt"] < _hlv \
                and (z.get("guc") or 0) >= 60:
            _yol = z
            break
    out["r_hedef"] = {
        "R": round(mesafe, 2), "kat": r_kat,
        "seviye": round(_hlv, 2),
        "getiri_pct": round(r_kat * mesafe / d1, 3) if d1 else None,
        "yol_direnci": ({"bant": f"{_yol['bant_alt']:,.2f}–"
                                 f"{_yol['bant_ust']:,.2f}",
                         "guc": _yol.get("guc"),
                         "not": ("Hedefe giden yolda güçlü direnç bölgesi — "
                                 "kısmi kâr alma noktası olarak "
                                 "değerlendirilebilir. Hedef yine 3R: "
                                 "walk-forward testte sabit R hedefi, "
                                 "direnç-bazlı hedefi yendi; bu yüzden "
                                 "hedef DİRENCE ÇEKİLMEDİ.")}
                        if _yol else None),
        "basabas_isabet": round(1.0 / (1.0 + r_kat), 3),
        "not": (f"Giriş {d1:,.2f} · stop {sec:,.2f} → 1R = {mesafe:,.2f}. "
                f"Hedef = giriş + {r_kat:.0f}R. "
                f"BAŞABAŞ İSABET: {r_kat:.0f}R hedefte sistem ancak "
                f"işlemlerin %{100/(1+r_kat):.0f}'inden fazlası kazanırsa "
                f"kâr eder (E = W×{r_kat:.0f}R − (1−W)×1R = 0 ⟹ "
                f"W = 1/(1+{r_kat:.0f})). Bu oranın altında sistem "
                f"MATEMATİKSEL OLARAK kaybeder. "
                f"⚠️ B7-2 DÜRÜSTLÜK UYARISI: bu notun eski hâli 'beklenti "
                f"+0.28R, isabet ~%15' diyordu — bu iki sayı saf 3R/1R "
                f"altında BİRLİKTE DOĞRU OLAMAZ (%15 isabet −0.40R verir; "
                f"+0.28R için %32 isabet gerekir). Rakamlar ancak '%15' TAM "
                f"HEDEFE ulaşma oranıysa ve kalan işlemlerin bir kısmı "
                f"trailing stop ile artıda kapanıyorsa uzlaşır. Kendi işlem "
                f"günlüğünle DOĞRULAMADAN bu beklentiye güvenme.")}
    portfoy = _num((a or {}).get("portfoy"))
    risk_pct = _num((a or {}).get("risk_pct")) or 1.0
    if mesafe > 0:
        hesap = None
        if portfoy and portfoy > 0:
            risk_usd = portfoy * risk_pct / 100.0
            adet = int(risk_usd // mesafe)
            hesap = {"risk_usd": round(risk_usd, 0), "hisse": adet,
                     "pozisyon_usd": round(adet * d1, 0),
                     "portfoy_pct": (round(adet * d1 / portfoy, 3)
                                     if portfoy else None)}
        out["pozisyon"] = {
            "kural": (f"Pozisyon = (portföy × %{risk_pct:.2g} risk) ÷ "
                      f"stop mesafesi (${mesafe:,.2f})"),
            "ornek_1000usd_risk": int(1000 // mesafe),
            "hesap": hesap,
            "not": ("%1-2 kuralı savunmacıdır (Tharp); optimal risk "
                    "stratejiye bağlıdır (Kelly) — hayatta kalma aracı, "
                    "getiri garantisi değil.")}
    out["rejim_notu"] = ("Kaminski-Lo (2014): stop, momentum rejiminde "
                         "değer katar; ortalamaya-dönüş rejiminde beklenen "
                         "getiriyi DÜŞÜREBİLİR. 3-4 aylık ABD hisse "
                         "ufkunda kullanımı savunulabilir, garanti değil.")
    out["gap_notu"] = ("Stop fiyat GARANTİSİ değildir: gap'te emir bir "
                       "sonraki fiyattan dolar (kazanç günü riski büyük). "
                       "Stop-market kesin dolar/fiyat belirsiz; stop-limit "
                       "fiyatı sınırlar ama hiç dolmayabilir.")
    return out


def seviye_kosullu_istatistik(hist, seviye, atr=None, teyit=1,
                              ufuklar=(5, 10, 21)):
    """
    SEVİYE KOŞULLU TARİHSEL İSTATİSTİK — "kapanış şu seviyenin ALTINDA /
    ÜSTÜNDE kalırsa ne olur?" sorusunun DÜRÜST cevabı: tahmin değil, bu
    hissenin KENDİ geçmişindeki benzer olayların baz oranları.
    Olay tanımı: kapanış seviyenin öbür tarafından gelir ve `teyit` ardışık
    kapanış boyunca yeni tarafta KALIR ("kalırsa" = teyit). Her ufuk için:
      medyan/p25/p75 ileri getiri · yön-devam oranı (ufuk sonunda kapanış
      olay yönünde) · geri-dönüş oranı (pencere içinde seviyenin öbür
      tarafına kapanış) · ek-derinleşme oranı (pencere ekstremi olay
      kapanışından ≥1×ATR öteye) · örneklem N.
    N küçükse sonuç çıkarılmaz — tablo N'i açıkça gösterir.
    """
    try:
        if hist is None or "Close" not in hist or not seviye:
            return None
        c = hist["Close"].dropna()
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        if len(c) < 60:
            return None
        atr = _num(atr) or float((c.pct_change().dropna().std() or 0.02)
                                 * c.iloc[-1])
        alt = (c < seviye).values
        ust = (c > seviye).values

        def olaylar(taraf):
            ev = []
            i = 1
            while i < len(c) - teyit:
                if not taraf[i - 1] and all(taraf[i:i + teyit]):
                    ev.append(i + teyit - 1)     # teyit tamamlanan bar
                    i += teyit
                else:
                    i += 1
            return ev

        def istat(evler, yon):
            out = {"n": len(evler), "ufuklar": {}}
            for h in ufuklar:
                rets, devam, geri, derin = [], 0, 0, 0
                for i in evler:
                    if i + h >= len(c):
                        continue
                    p0 = float(c.iloc[i])
                    seg = c.iloc[i + 1:i + 1 + h]
                    rets.append(float(seg.iloc[-1]) / p0 - 1)
                    if yon == "alt":
                        devam += int(float(seg.iloc[-1]) < p0)
                        geri += int(bool((seg > seviye).any()))
                        derin += int(float(seg.min()) <= p0 - atr)
                    else:
                        devam += int(float(seg.iloc[-1]) > p0)
                        geri += int(bool((seg < seviye).any()))
                        derin += int(float(seg.max()) >= p0 + atr)
                if rets:
                    s = pd.Series(rets)
                    out["ufuklar"][f"{h}g"] = {
                        "n": len(rets),
                        "medyan": round(float(s.median()), 4),
                        "p25": round(float(s.quantile(.25)), 4),
                        "p75": round(float(s.quantile(.75)), 4),
                        "yon_devam": round(devam / len(rets), 2),
                        "geri_donus": round(geri / len(rets), 2),
                        "ek_derinlesme_1atr": round(derin / len(rets), 2)}
            return out

        return {"seviye": round(float(seviye), 2), "teyit_kapanis": teyit,
                "altina_kirilma": istat(olaylar(alt), "alt"),
                "ustune_gecis": istat(olaylar(ust), "ust"),
                "not": ("Bu hissenin KENDİ geçmişinden baz oranlar — "
                        "olasılık eğilimi, GARANTİ DEĞİL. N küçükse "
                        "(<8) sonuç çıkarma; gelecek geçmişe benzemek "
                        "zorunda değildir.")}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_saatlik(ticker: str):
    """1 saatlik barlar (Yahoo sınırı: ~730 gün, ~15 dk gecikme).
    4 saatlik analiz bunlardan türetilir; hata → None."""
    try:
        import yfinance as yf  # FIX: sessiz NameError tamiri
        tk = yf.Ticker(ticker, session=_yf_session())
        # B3-6: Yahoo'nun 1h sınırı TAM 730 gündür; tavanın tam üstünde
        # istek, sınır hesabı saat diliminden kaydığında ARALIKLI hata verir.
        h = _yf_retry(lambda: tk.history(period="725d", interval="1h"))
        if h is None or getattr(h, "empty", True):
            return None
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        return h
    except Exception:
        return None


def to_4h(h1):
    """1 saatlik barları 4'erli gruplayıp 4 SAATLİK bar üretir (işlem-barı
    bazlı: seans boşluklarından etkilenmez). OHLCV korunur."""
    try:
        # B3-3: ABD seansı 09:30-16:00 = 6.5 saat → günde 7 saatlik bar.
        # `np.arange//4` bar SAYARAK gruplar ve seansla HİÇ hizalanmaz:
        # ölçümde barların %40'ı iki farklı günü birleştiriyor ve GECELİK
        # BOŞLUĞU bar-içi hareket sayıyordu → ATR 1.3 KAT şişik, bu da
        # doğrudan stop mesafesine ve pozisyon boyutuna giriyor.
        # ARTIK: gruplama her seans İÇİNDE, gün sınırında sıfırlanır.
        kolon = [k for k in ("Open", "High", "Low", "Close", "Volume")
                 if k in h1.columns]
        c = h1[kolon].dropna(subset=["Close"]).sort_index()
        if len(c) < 8:
            return None
        satir = []
        for _gun, blok in c.groupby(c.index.normalize()):
            blok = blok.sort_index()
            for i in range(0, len(blok), 4):
                d = blok.iloc[i:i + 4]
                if d.empty:
                    continue
                kayit = {"ts": d.index[-1], "bar_sayisi": len(d)}
                if "Open" in kolon:
                    kayit["Open"] = float(d["Open"].iloc[0])
                if "High" in kolon:
                    kayit["High"] = float(d["High"].max())
                if "Low" in kolon:
                    kayit["Low"] = float(d["Low"].min())
                kayit["Close"] = float(d["Close"].iloc[-1])
                if "Volume" in kolon:
                    kayit["Volume"] = float(d["Volume"].sum())
                satir.append(kayit)
        if len(satir) < 8:
            return None
        return pd.DataFrame(satir).set_index("ts")
    except Exception:
        return None


def momentum_12_1(hist):
    """
    12-1 MOMENTUM (Jegadeesh-Titman 1993, J. of Finance 48/1): son 12 ayın
    getirisi, SON 1 AY HARİÇ (kısa vadeli geri dönüş etkisi ayıklanır).
    12 ay formasyon / 3 ay tutma kombinasyonu, kullanıcının 3 aylık
    ufkuyla literatürde en iyi örtüşen kanıtlı sinyaldir (~%1.3/ay,
    1965-1989, NYSE/AMEX). Formül: P[t−21] / P[t−252] − 1
    B3-1 SONRASI NOT: fetch_bundle artık İKİ baz üretir —
    b['hist'] bölünme düzeltmeli/TEMETTÜ DÜZELTMESİZ (işlem
    seviyeleri), b['hist_tr'] ise toplam getiri. Momentum bir
    GETİRİ ölçüsü olduğu için TOPLAM GETİRİ serisiyle beslenmelidir
    (literatür momentumu toplam getiriyle tanımlar).
    YÖN GARANTİSİ DEĞİLDİR; SKORA GİRMEZ — skor fiyat-değer sorusudur,
    bu bir ZAMANLAMA bağlamıdır. Yalnız kademeli girişin temposunu
    yavaşlatan bir bayrak üretir (bütünsel rapor G4).
    """
    try:
        if hist is None or "Close" not in hist:
            return None
        c = hist["Close"].dropna()
        if len(c) < 253:
            return None
        mom = float(c.iloc[-22]) / float(c.iloc[-253]) - 1.0
        # ── B7-1 (YAPISAL HATA) ───────────────────────────────────────
        # Jegadeesh-Titman momentumu KESİTSELDİR: hisseleri birbirine göre
        # sıralar, üst desili alır ALT DESİLİ SATAR. Etki, sıralı bir
        # evrenin uçları arasındaki FARKTA yaşar — tek bir hissenin kendi
        # işaretinde DEĞİL. Moskowitz-Ooi-Pedersen (2012) zaman-serisi
        # momentumunu 58 VADELİ SÖZLEŞMEDE buldu, tek hisselerde değil;
        # Lim-Wang-Yao (2018) tek hisselerde alfanın kesitsel momentum
        # kontrol edilince ANLAMSIZLAŞTIĞINI, Goyal-Jegadeesh (2018) farkın
        # "zamanla değişen net-uzun yatırım" (piyasa betası) olduğunu
        # gösterdi. Bu yüzden burada SİNYAL ÜRETİLMEZ.
        return {"getiri_12_1": round(mom, 4),
                "yon": "POZİTİF" if mom > 0 else "NEGATİF",
                "sinyal": None,
                "not": ("⚠️ Bu YALNIZCA TREND BAĞLAMIDIR — belgelenmiş bir "
                        "sinyal DEĞİLDİR. Momentum KESİTSELDİR; tek bir "
                        "hissenin 12-1 getirisinin işareti, o hissenin kendi "
                        "gelecek getirisi için hakemli literatürde sağlam "
                        "öngörü gücü TAŞIMAZ (Lim-Wang-Yao 2018; "
                        "Goyal-Jegadeesh 2018; Huang vd. 2020: 55 varlığın "
                        "47'sinde t<1.65). Tempo kararını buna dayandırma.")}
    except Exception:
        return None


def pead_sinyali(cat):
    """
    PEAD — KAZANÇ SÜRPRİZİ SONRASI SÜRÜKLENME (Ball-Brown 1968;
    Bernard-Thomas 1989/1990): sürpriz sonrası fiyat haftalar-aylar
    boyunca AYNI yönde sürüklenme eğilimindedir (~60 işlem günü); 3 aylık
    ufka uyan en kanıtlı ikinci sinyaldir (bütünsel rapor G5).
    Girdi: parse_catalysts'in kazanç sürpriz geçmişi (yfinance
    earnings_dates → Surprise(%); tarihe göre ARTAN sıralı gelir).
    SUE-lite: son sürpriz ÷ son sürprizlerin std'si (≥3 örneklemde).
    DÜRÜSTLÜK: etki büyük-cap'te zayıflamıştır; yön garantisi değildir;
    SKORA GİRMEZ — kademeli girişin temposunu yönetir.
    """
    try:
        ges = [g for g in ((cat or {}).get("kazanc_surpriz_gecmisi") or [])
               if _num(g.get("surpriz")) is not None and g.get("tarih")]
        if not ges:
            return None
        son = ges[-1]                          # liste artan tarihli
        s = float(son["surpriz"])
        try:
            gun = int((pd.Timestamp.now().normalize()
                       - pd.Timestamp(son["tarih"])).days)
        except Exception:
            gun = None
        sue = None
        if len(ges) >= 3:
            import statistics
            _sd = statistics.pstdev(float(g["surpriz"]) for g in ges)
            sue = round(s / _sd, 2) if _sd > 1e-9 else None
        yon = "POZİTİF" if s > 2 else ("NEGATİF" if s < -2 else "NÖTR")
        ardisik = 0
        for g in reversed(ges):
            gs = float(g["surpriz"])
            if gs != 0 and (gs > 0) == (s > 0):
                ardisik += 1
            else:
                break
        pencerede = (gun is not None and 0 <= gun <= 90)
        return {"son_surpriz_pct": round(s, 2), "yon": yon,
                "gun_once": gun, "suruklenme_penceresinde": pencerede,
                "sue_yaklasik": sue, "ardisik_ayni_yon": ardisik,
                "not": ("PEAD (Bernard-Thomas 1989): sürpriz sonrası ~60 "
                        "işlem günü sürüklenme. Pencere içinde NEGATİF "
                        "sürpriz → yeni dilim açmakta acele etme; POZİTİF "
                        "sürpriz → 3 aylık tez için destekleyici rüzgâr. "
                        "KANIT SINIRI: Martineau (2022, Critical Finance "
                        "Review) büyük-cap hisselerde PEAD'in ~2006'dan beri "
                        "İSTATİSTİKSEL OLARAK YOK olduğunu buldu; etki "
                        "küçük/orta-cap ve düşük analist kapsamlı hisselerde "
                        "kalmıştır. Büyük-cap'te bu satırı SİNYAL SAYMA. "
                        "Yön garantisi değildir; skora GİRMEZ.")}
    except Exception:
        return None


def kademeli_plan(m):
    """
    KADEMELİ GİRİŞ / BEKLEME PLANI iskeleti — TAVSİYE DEĞİL, disiplin
    çerçevesi. İlke: asla tek seferde tam pozisyon; fiyat cazipse kademeli
    gir, pahalıysa plan YOK, sadece izleme eşiği. Claude katmanı bu iskeleti
    katalizör takvimi ve falsifikasyonla rafine eder.

    ÇIPA MANTIĞI (kritik düzeltme): Seviyeler önce MEVCUT FİYATTAN, pahalılık
    skoruna orantılı iskontoyla türetilir. DCF adil değeri YALNIZCA makul
    banttaysa (fiyatın %25–%300'ü) ikinci bir çapraz-kontrol olarak katılır;
    absürtse tamamen yok sayılır. Böylece izleme eşiği HER ZAMAN fiyata göre
    anlamlıdır — asla uçuk bir DCF sayısına düşmez.
    """
    price, fair, score = m.get("price"), m.get("fair"), m.get("score")
    if not price or score is None:
        return None
    fair_ok = _fair_makul_mu(fair, price)
    mc = m.get("mc") or {}
    # Değer bağlamı: dilim/eşik seviyelerinin yanında, makul banttaki
    # bağımsız değer çıpaları da gösterilir (şeffaf çapraz kontrol).
    _bag = {}
    for _ad, _v in (("dcf_baz", fair if fair_ok else None),
                    ("mc_p25", mc.get("p25")), ("mc_p50", mc.get("p50"))):
        _bag[_ad] = (round(float(_v), 2)
                     if _v and _fair_makul_mu(_v, price) else None)
    deger_baglami = _bag if any(_bag.values()) else None

    yak = [e for e in (m.get("catalysts") or {}).get("gelecek", [])
           if 0 <= e.get("gun_kaldi", 999) <= 21]
    olay_notu = (f"⏳ {yak[0]['gun_kaldi']} gün sonra '{yak[0]['olay']}' var — "
                 f"ilk dilimden önce bu olayı görmek disiplinli olur."
                 if yak else None)

    # ── REJİM TEMPOSU (rapor G4/G5/G10): momentum / PEAD / MA200 planın
    # SEVİYELERİNİ değiştirmez, TEMPOSUNU değiştirir. Kanıt: Jegadeesh-
    # Titman 1993 (12ay/3ay); Bernard-Thomas 1989 (PEAD ~60 işlem günü).
    # Ayı rejiminde kademeli giriş "değer tuzağına kademeli giriş"
    # olabilir — seviyeler kalır, acele gider.
    tempo = []
    # B7-1: mutlak momentum işareti belgelenmiş bir sinyal olmadığı için
    # artık TEMPO KARARI VERMİYOR; yalnız bağlam olarak not düşülüyor.
    if (m.get("momentum") or {}).get("yon") == "NEGATİF":
        tempo.append("12-1 getirisi negatif — bu YALNIZCA trend bağlamıdır, "
                     "belgelenmiş bir sinyal DEĞİLDİR (momentum kesitseldir; "
                     "tek hissede zaman-serisi momentumu literatürde "
                     "doğrulanmamıştır). Tempo kararını buna dayandırma.")
    _pp = m.get("pead") or {}
    if _pp.get("yon") == "NEGATİF" and _pp.get("suruklenme_penceresinde"):
        tempo.append(f"Negatif kazanç sürprizi "
                     f"(%{_pp['son_surpriz_pct']:+.1f}, "
                     f"{_pp['gun_once']}g önce) sürüklenme penceresinde — "
                     f"PEAD aleyhte: pencere (~90g) kapanana dek YENİ "
                     f"DİLİM AÇMA (Bernard-Thomas 1989).")
    elif _pp.get("yon") == "POZİTİF" and _pp.get("suruklenme_penceresinde"):
        tempo.append("Pozitif kazanç sürprizi sürüklenmesi LEHTE (PEAD) — "
                     "3 aylık ufuk için destekleyici rüzgâr; dilim "
                     "disiplini yine de değişmez.")
    try:
        _hc_ = m.get("_hist")
        _hc_ = _hc_["Close"].dropna() if (_hc_ is not None
                                          and "Close" in _hc_) else None
        if (_hc_ is not None and len(_hc_) >= 200 and price
                and price < float(_hc_.rolling(200).mean().iloc[-1])):
            tempo.append("Fiyat MA200 ALTINDA (ayı rejimi bağlamı) — dilim "
                         "aralıklarını genişlet, stop disiplinini sıkı tut "
                         "(rapor G10); ucuzlayan daha da ucuzlayabilir.")
    except Exception:
        pass

    def lvl(x):
        return round(x, 2) if x else None

    # ── MERDİVEN (yeniden kalibre edildi) ──────────────────────────
    # Eski eşik 55'ti; testte kaliteli hisselerin skoru medyan 63-79
    # çıktığı için plan HİÇ kurulmuyordu. Yeni eşik 75. Ayrıca "cazip"
    # eşiği 35→45'e çekildi ve mevcut fiyattan alım kaldırıldı (CVX'te
    # dilim1 = fiyat çıkmıştı: iskonto yok, sabır yok).
    if score >= 75:
        # PAHALI: izleme eşiği = fiyattan skorla orantılı iskonto.
        # DENETİM BULGUSU K: eski yorum "skor 55→%12, 97→~%38" diyordu ama bu
        # dal 75'ten BAŞLIYOR; gerçek aralık skor 75→%10, 85→%21, 97→%35.
        # Yorum eski eşikten kalmıştı, düzeltildi.
        # DCF makulse ve daha muhafazakârsa ikisinden DÜŞÜK olanı al.
        ESIK_MAX_ISKONTO = 0.35          # bu dalın tasarım tavanı
        isk = min(0.10 + (score - 75) / 22 * 0.25, ESIK_MAX_ISKONTO)
        esik_fiyat = price * (1 - isk)
        kaynak = f"fiyattan ~%{isk*100:.0f} iskonto"
        esik_taban_notu = None
        if fair_ok and fair < esik_fiyat:
            # DENETİM BULGUSU J (YÜKSEK): burada ULAŞILABİLİRLİK TABANI YOKTU.
            # _fair_makul_mu bandı çok geniş (fiyatın %25-%300'ü), dolayısıyla
            # DCF eşiği fiyatın %70 altına çakabiliyordu — kullanıcının zaten
            # şikayet ettiği "JNJ 160 → 105, fiyat oraya hiç inmiyor" hatasının
            # ta kendisi. O hata 45-75 dalında DILIM_MAX_ISKONTO ile çözülmüş
            # ama BU dal atlanmıştı. Artık DCF eşiği daha muhafazakâr yapabilir
            # AMA dalın tasarım tavanını (%35) aşamaz; aşan bilgi kaybolmuyor,
            # nota yazılıyor.
            _taban = price * (1 - ESIK_MAX_ISKONTO)
            if fair < _taban:
                esik_fiyat = _taban
                kaynak = (f"fiyattan %{ESIK_MAX_ISKONTO*100:.0f} iskonto "
                          f"(tavan) — DCF daha da düşüktü")
                esik_taban_notu = (
                    f"DCF adil değeri ({fair:,.2f}) fiyatın "
                    f"%{ESIK_MAX_ISKONTO*100:.0f} altındaki tabandan da düşük. "
                    f"Eşik tabana ({esik_fiyat:,.2f}) sabitlendi — daha aşağısı "
                    f"bu ufukta ulaşılamaz olurdu ve plan değil DUVAR olurdu. "
                    f"Adil değerin bu kadar düşük olması ŞİRKETİN ÇOK PAHALI "
                    f"olduğu anlamına gelir; eşik geldi diye otomatik alma.")
            else:
                esik_fiyat = fair
                kaynak = "DCF adil değeri (makul bantta, daha muhafazakâr)"
        elif fair is not None and not fair_ok:
            kaynak += "; DCF adil değeri absürt bantta olduğu için YOK SAYILDI"
        # CANLI BULGU: bu dalda eşik ÇIPLAK YÜZDEDİR — destek/direnç motoru
        # hiç kullanılmıyordu. INTC'de eşik 66.37 çıkarken gerçek destekler
        # 45.7-52.2 idi: ekrandaki sayı grafikte hiçbir şeye denk gelmiyordu.
        # Seviyeyi DEĞİŞTİRMİYORUZ (tasarım kararı); yanına yapıyı koyuyoruz.
        _pd = [z for z in ((m.get("sr") or {}).get("destekler") or [])]
        _yakin = next((z for z in _pd if z["bant_ust"] <= esik_fiyat), None)
        _yapisal_not = None
        if _pd:
            _en_yakin = _pd[0]
            _yapisal_not = (
                f"Grafikteki en yakın gerçek destek bölgesi "
                f"{_en_yakin['bant_alt']:,.2f}–{_en_yakin['bant_ust']:,.2f} "
                f"(fiyatın %{(1 - _en_yakin['bant_ust']/price)*100:.0f} "
                f"altı). İzleme eşiği ({esik_fiyat:,.2f}) bir DESTEK "
                f"SEVİYESİ DEĞİLDİR — skordan türetilmiş bir iskonto "
                f"çıpasıdır."
                + (f" Eşiğin altında kalan ilk yapısal bölge: "
                   f"{_yakin['bant_alt']:,.2f}–{_yakin['bant_ust']:,.2f}."
                   if _yakin else
                   " Eşiğin altında kayıtlı yapısal bölge yok."))
        return {"mod": "PLAN_YOK_IZLE",
                "esik_taban_notu": esik_taban_notu,
                "yapisal_baglam": _yapisal_not,
                "esik_hareketli_uyarisi": (
                    "İZLEME EŞİĞİ SABİT BİR HEDEF DEĞİLDİR: her çalıştırmada "
                    "O ANKİ fiyattan yeniden hesaplanır. Fiyat düşerse eşik "
                    "de düşer. Alım dilimleri ancak pahalılık skoru 75'in "
                    "ALTINA inince üretilir — bu, fiyat düşüşüyle otomatik "
                    "olmaz, temellerin de (marj, FCF, çarpanlar) değişmesi "
                    "gerekebilir."),
                "yontem_sinirlari": (
                    "KANIT SINIRI: değer sinyalleri uzun ufuklu (1-5 yıl) "
                    "fenomenlerdir; 3-4 aylık pencerede sinyal/gürültü oranı "
                    "düşüktür ve etki büyük-cap'te zayıflar "
                    "(Lakonishok-Shleifer-Vishny 1994; Israel-Moskowitz "
                    "2013; Arnott ve ark. 2021). 'Pahalı' hükmü bu ufukta "
                    "bir ZAMANLAMA kehaneti değil, RİSK BÜTÇESİ kararıdır: "
                    "eşiğe gelmeden almamak yerine daha KÜÇÜK pozisyon "
                    "almak da savunulabilir bir okumadır."),
                "mesaj": "Fiyat pahalı bölgede — kademeli plan YOK. Harika "
                         "şirketi yanlış fiyattan almak en pahalı hatadır; "
                         "izleme eşiğine gelmeden alım tartışılmaz.",
                "izleme_esigi": lvl(esik_fiyat),
                "esik_kaynagi": kaynak,
                "deger_baglami": deger_baglami,
                "olay_notu": olay_notu,
                "rejim_temposu": tempo or None,
                "kurallar": ["Eşik gelmeden alım yok",
                             "Tez koşulları değişirse eşik yeniden "
                             "hesaplanır"]}

    # CAZİP/MAKUL: ilk dilim çıpası. Fiyat-önce; DCF makulse çapraz-kontrol.
    _fair_ezdi = False                        # her iki dalda tanımlı olmalı
    if score < 45:
        b1 = price * 0.98                     # cazip ama yine de tampon
        mesaj = ("Fiyat cazip bölgede: plan, mevcut fiyattan başlayan 3 "
                 "dilimdir. Her dilim ~1/3 — düşüş dilim atlatmaz, dilim "
                 "DOLDURUR.")
        mod = "KADEMELİ_GİRİŞ"
    else:
        # makul (45–75): iskonto skorla orantılı — 45'te %3, 75'te %10.
        # Sabit %3 yerine kademeli: pahalılaştıkça daha çok sabır.
        b1 = price * (1 - (0.03 + (score - 45) / 30 * 0.07))
        # ULAŞILABİLİRLİK TABANI (kritik düzeltme):
        # "DCF makul bantta ise dilimi o belirlesin" kuralı, kaliteli
        # hisselerde dilimi fiyatın %30-50 altına çakıyordu. Fiyat oraya
        # hiç inmediği için plan hiç tetiklenmiyordu — plan değil DUVAR.
        # Artık dilim fiyattan EN FAZLA %20 aşağı olabilir. Adil değer
        # daha da düşükse bu bilgi kaybolmaz: 'deger_baglami' ve
        # 'dilim_taban_notu' alanlarında görünür.
        taban = price * (1 - DILIM_MAX_ISKONTO)
        if fair_ok and fair < b1:
            b1 = max(fair, taban)
            _fair_ezdi = fair < taban
        # DENETİM BULGUSU F: üç modül "pahalı"yı üç farklı eşikle tanımlıyor —
        # verdict_info 55, cikis_plani 55, kademeli_plan 75. Sonuç: skor 65'te
        # program aynı anda "PAHALI" + "KÂR ALMA BÖLGESİ" + "işte alım
        # dilimlerin" diyordu ve bu mesaj "Fiyat makul" diye BAŞLIYORDU.
        # 75 eşiği bilinçli kalibrasyondur (55'te kaliteli hisselerde plan hiç
        # kurulmuyordu) — DEĞİŞTİRİLMEDİ. Düzeltilen şey mesajın dürüstlüğü:
        # 55 üstünde bu bir ALIM çağrısı değil BEKLEME merdivenidir.
        mesaj = (("Hüküm PAHALI bölgede — bu merdiven bir ALIM çağrısı "
                  "DEĞİL, BEKLEME merdivenidir: dilimler ancak fiyat "
                  "belirgin gerilerse devreye girer, bugünden alım "
                  "tartışılmaz. " if score >= 55 else
                  "Fiyat makul ama iskonto yok: ilk dilim, fiyat hafif "
                  "gerilediğinde (ya da makulse baz adil değere indiğinde). ")
                 + "Aceleye gerek yok — kenar, sabırdadır.")
        mod = "SINIRDA_BEKLE"
    # ── DİLİMLERİ DESTEK BÖLGESİNE ÇIPALA ──────────────────────────
    # Yüzde-iskontolu dilim KEYFÎDİR ve fiyatla birlikte aşağı kayar
    # (fiyat 125→118 olunca dilim 117→110 oluyordu; hedef kaçıyordu).
    # Gerçek destek bölgesi tarihsel bir yapıdır: fiyat oynasa da yeri
    # DEĞİŞMEZ. Bu yüzden b1'e yakın bir destek bölgesi varsa dilim
    # oraya çakılır ve "yapısal" diye etiketlenir. Yoksa yüzde dilimi
    # kullanılır ama "yapısal dayanağı YOK" uyarısıyla.
    _dz = [z for z in ((m.get("sr") or {}).get("destekler") or [])
           if z.get("bant_ust") and z["bant_ust"] <= price * 1.005]
    _dz = sorted(_dz, key=lambda z: -z["bant_ust"])
    # Dilimler SIRAYLA fiyatın altındaki bölgelere oturur: 1. dilim en
    # yakın bölge, 2. dilim bir alttaki, 3. dilim onun altındaki. Bölge
    # fiyata göre değil TARİHE göre sabit olduğundan, fiyat oynasa da
    # dilim yerinde kalır — asıl çözmek istediğimiz sorun buydu.
    # Çok uzaktaki bölgeler (fiyatın %20'sinden aşağı) 1. dilim olamaz;
    # o durumda yüzde dilimi kullanılır (ulaşılabilirlik tabanı).
    # ── DİLİM YERLEŞTİRME (iki hata düzeltildi) ────────────────────
    # D1 (SIRA BOZULMASI): eski kod, bir önceki dilim "yüzde" ise
    # _yedek[_i]'ye (b1 çıpalı sabit merdiven) dönüyordu. Tek uygun bölge
    # olan durumda sonuç: dilim3 > dilim1 çıkıyordu (ör. 85 · 78.2 · 82.3)
    # — yani 3. dilim 1. dilimin ÜSTÜNDE, merdiven ters dönüyordu.
    # DÜZELTME: bölge kalmadığında her zaman BİR ÖNCEKİ dilimin altından
    # devam edilir → merdiven her senaryoda kesin azalandır.
    # D2 (DERİN BÖLGELER ÇÖPE GİDİYORDU): ulaşılabilirlik tabanı
    # (fiyatın %20 altı) TÜM dilimlere uygulanıyordu. Oysa kodun kendi
    # gerekçesi "1. dilim olamaz" diyor. Sonuç: 78'deki GERÇEK destek
    # elenip yerine 75.4'te UYDURMA yüzde dilimi konuyordu — hem daha
    # aşağı hem yapısal dayanaksız. DÜZELTME: taban yalnız 1. dilime
    # uygulanır; 2-3. dilimler sıradaki gerçek bölgeleri kullanır.
    # Dipsiz kuyuya inmemek için derin sınır, pahalı daldaki aynı
    # ulaşılabilirlik tavanıdır (%35); ötesi yüzde merdivenine düşer.
    _taban_f = price * (1 - DILIM_MAX_ISKONTO)     # 1. dilim tabanı
    _derin_taban = price * (1 - 0.35)              # 2-3. dilim sınırı
    _yedek = [b1, b1 * 0.92, b1 * 0.84]
    _havuz = list(_dz)                             # fiyat altı, azalan
    _yerlesmis = []
    for _i in range(3):
        _esik = _taban_f if _i == 0 else _derin_taban
        z = None
        while _havuz:
            aday = _havuz[0]
            # ALIM TURU BULGUSU E2: 1. dilim, skora göre hesaplanan iskonto
            # çıpasının (b1) belirgin ÜSTÜNDE bir bölgeye oturamaz — yoksa
            # fiyatın %1 altındaki herhangi bir bölge, "BEKLE" dalındaki
            # %3-10 sabır iskontosunu sessizce deliyordu (skor 60'ta dilim1
            # 99.2 çıkıyordu; b1=93.5 olmalıydı). %2 tolerans: b1'e YAKIN
            # bölge yine yapısal çıpadır; uzaktaki üst bölgeler atılır.
            if _i == 0 and aday["bant_ust"] > b1 * 1.02:
                _havuz.pop(0)
                continue
            if _yerlesmis and aday["bant_ust"] >= _yerlesmis[-1]["seviye"]:
                _havuz.pop(0)                      # üstte kalanı atla
                continue
            if aday["bant_ust"] < _esik:
                break                              # çok derin → yüzdeye geç
            z = _havuz.pop(0)
            break
        if z is not None:
            _yerlesmis.append({
                "seviye": lvl(z["bant_ust"]), "cipa": "destek bölgesi",
                "bolge": f"{z['bant_alt']:,.2f}–{z['bant_ust']:,.2f}",
                "guc": z.get("guc"), "bant_alt": z["bant_alt"]})
        else:
            _ref = (_yerlesmis[-1]["seviye"] * 0.92 if _yerlesmis
                    else _yedek[_i])
            _yerlesmis.append({"seviye": lvl(_ref), "cipa": "yüzde",
                               "bolge": None, "guc": None,
                               "bant_alt": None})
    dilimler = [{"dilim": i + 1, "seviye": x["seviye"], "oran": "~1/3",
                 "cipa": x["cipa"], "bolge": x["bolge"], "guc": x["guc"],
                 "bant_alt": x["bant_alt"]}
                for i, x in enumerate(_yerlesmis)]
    _yapisal = sum(1 for x in _yerlesmis if x["cipa"] == "destek bölgesi")
    cipa_kalitesi = {
        "yapisal_dilim": _yapisal, "toplam": len(_yerlesmis),
        "not": ("Dilimlerin " + ("HEPSİ" if _yapisal == 3
                                 else f"{_yapisal}/3'ü")
                + " gerçek destek bölgesine çıpalı — bu seviyeler fiyat "
                  "oynasa da YERİNDE KALIR."
                if _yapisal else
                "HİÇBİR dilim destek bölgesine denk gelmiyor: seviyeler "
                "yalnız yüzde iskontodan türetildi. Bu keyfîdir ve fiyat "
                "düştükçe aşağı kayar — yapısal dayanağı olmayan bir "
                "dilime emir koymak zayıf bir plandır.")}
    # Giriş dilimlerinde de çapraz teyit: ±%8 bandına düşen değer çıpaları.
    giris_cipalari = [("DCF baz", fair if fair_ok else None),
                      ("MC P25", mc.get("p25")), ("MC P50", mc.get("p50"))]
    giris_cipalari = [(ad, float(v)) for ad, v in giris_cipalari
                      if v and _fair_makul_mu(v, price)]
    sr_z = (m.get("sr") or {}).get("destekler") or []
    for d_ in dilimler:
        sv = d_["seviye"]
        yak = [ad for ad, v in giris_cipalari
               if sv and abs(v - sv) / sv <= 0.08]
        for z in sr_z:                       # fiyat yapısı çapraz-teyidi
            if sv and z["bant_alt"] * 0.99 <= sv <= z["bant_ust"] * 1.01:
                yak.append(f"Destek bölgesi (güç {z['guc']:.0f})")
                break
        d_["teyit"] = ", ".join(yak) if yak else None
    taban_notu = None
    if _fair_ezdi:
        taban_notu = (
            f"DCF adil değeri ({fair:,.2f}) fiyatın %{DILIM_MAX_ISKONTO*100:.0f}"
            f" altındaki tabandan da düşük. Dilim tabana ({b1:,.2f}) "
            f"sabitlendi — daha aşağısı bu ufukta ulaşılamaz olurdu. "
            f"Adil değerin bu kadar düşük olması ŞİRKETİN PAHALI olduğu "
            f"anlamına gelir; dilim geldi diye otomatik alma.")
    return {"mod": mod, "mesaj": mesaj, "deger_baglami": deger_baglami,
            "dilimler": dilimler, "dilim_taban_notu": taban_notu,
            "cipa_kalitesi": cipa_kalitesi,
            "cipa_notu": ("Seviyeler mevcut fiyata çıpalı"
                          + ("; DCF adil değeri makul bantta olduğu için "
                             "çapraz-kontrol edildi." if fair_ok else
                             "; DCF adil değeri absürt/eksik olduğu için "
                             "yalnızca fiyat kullanıldı.")),
            "olay_notu": olay_notu,
            "rejim_temposu": tempo or None,
            "yontem_durustlugu": (
                "KADEMELİ GİRİŞ BİR GETİRİ OPTİMİZASYONU DEĞİLDİR. Kanıt: "
                "toplu giriş, kademeli girişi tarihsel olarak vakaların "
                "~2/3'ünde ve ortalama ~2,3 puan geçer (Vanguard 2012; "
                "Constantinides 1979). Kademeli giriş beklenen getiriyi "
                "DÜŞÜRÜR, karşılığında düşüş riskini ve yanlış-zamanlama "
                "pişmanlığını azaltır. Bu bir DAVRANIŞSAL SİGORTADIR; "
                "bilerek ödenen bir primdir."),
            "kurallar": ["Asla tek seferde tam pozisyon alma",
                         "Tez bozulursa (falsifikasyon koşulları) kalan "
                         "dilimler İPTAL",
                         "STOP KIRILIRSA kalan dilimler İPTAL — stopun "
                         "altında dilim doldurmak stop kuralını geçersiz "
                         "kılar (aynı pozisyonda hem 'düşerken ekle' hem "
                         "'düşerse çık' mantıksal olarak tutarsızdır; "
                         "Kaminski-Lo 2014)",
                         "Dilim seviyesi geldi diye otomatik alma — o günkü "
                         "habere bak"]}


def cikis_plani(m, maliyet=None):
    """
    KADEMELİ ÇIKIŞ / SATIŞ PLANI — girişin aynası; TAVSİYE DEĞİL, disiplin
    çerçevesi. Üç bacak:
      1) DEĞER HEDEFLERİ: kademeli kâr alma seviyeleri. Çıpalar makul-bant
         süzgecinden geçer (fiyatın %25-%300'ü) — TTWO dersi: absürt DCF
         sayısı plana ASLA sızamaz; çıpa yoksa fiyat-primli merdiven.
      2) TEZ-BOZAN TETİKLER: fiyattan bağımsız çıkış koşulları (en güçlü
         satış kuralı fiyat hedefi değil, tezin kırılmasıdır).
      3) ZAMAN STOPU: katalizörler boş geçerse ölü parayı taşıma.
    Maliyet verilirse bağlam gösterilir ama karar çıpası YAPILMAZ —
    'başa baş gelsin de satayım' (disposition effect) tuzağına açık uyarı.
    """
    price, score = m.get("price"), m.get("score")
    if not price or score is None:
        return None
    mc = m.get("mc") or {}

    # 1) Değer çıpaları (makul-bant süzgeci + fiyatın anlamlı üstü)
    adaylar = []
    esik_alt = price * (0.98 if score >= 55 else 1.03)
    if score >= 55:                       # pahalıda kâr alma buradan başlar
        adaylar.append(("mevcut fiyat (pahalı bölge)", float(price)))
    analist_hedef = (((m.get("revizyon") or {}).get("hedef_fiyat") or {})
                     .get("ortalama"))
    for ad, v in (("DCF baz", m.get("fair")),
                  ("Monte Carlo P75", mc.get("p75")),
                  ("DCF iyimser", m.get("fair_hi")),
                  ("Monte Carlo P95", mc.get("p95")),
                  ("Analist hedef (ort.)", analist_hedef)):
        if v and _fair_makul_mu(v, price) and v > price * 1.03:
            adaylar.append((ad, float(v)))
    if score < 35:
        primler = (1.18, 1.32, 1.48)
    elif score < 55:
        primler = (1.12, 1.22, 1.34)
    else:
        primler = (1.00, 1.08, 1.16)
    for p in primler:
        adaylar.append((f"fiyat +%{int(round((p - 1) * 100))}",
                        round(price * p, 2)))
    hedefler = []
    deger_cipali = False
    for ad, v in sorted(adaylar, key=lambda x: x[1]):
        if v < esik_alt:
            continue
        if hedefler and v <= hedefler[-1]["seviye"] * 1.05:
            continue
        hedefler.append({"dilim": len(hedefler) + 1,
                         "seviye": round(v, 2), "kaynak": ad,
                         "oran": "~1/3"})
        if not ad.startswith("fiyat") and "mevcut" not in ad:
            deger_cipali = True
        if len(hedefler) == 3:
            break
    if not hedefler:
        return None
    # ── DENETİM BULGUSU L (YÜKSEK): DEĞER ÇIPALARI EZİLİYORDU ─────────
    # Adaylar ARTAN sıralanıp ilk 3'ü alınıyordu. Fiyat-primi merdiveni
    # (pahalı dalda +%0/+8/+16) her zaman değer çıpalarından DAHA DÜŞÜK
    # olduğu için ilk üç yeri kapıyor, DCF/Monte Carlo/analist hedeflerinin
    # HİÇBİRİ merdivene giremiyordu. Test: fiyat 100'de DCF 130, MC P75 140,
    # analist 135 varken merdiven 100/108/116 çıkıyordu — beş değer çıpası
    # hesaplanıp çöpe gidiyordu ve 'deger_cipali' hep False kalıyordu.
    # DÜZELTME: hiç değer çıpası giremediyse ve merdivenin ÜSTÜNDE geçerli
    # bir değer çıpası varsa, SON dilim onunla değiştirilir. Böylece plan
    # hem yakın-vade disiplinini (ilk dilimler) hem değer tabanlı üst hedefi
    # korur. Değiştirilen dilim açıkça etiketlenir.
    _deger_ust = sorted(
        (v, ad) for ad, v in adaylar
        if not ad.startswith("fiyat") and "mevcut" not in ad
        and v > hedefler[-1]["seviye"] * 1.05)
    if not deger_cipali and _deger_ust:
        _v, _ad = _deger_ust[0]
        hedefler[-1] = {"dilim": hedefler[-1]["dilim"],
                        "seviye": round(_v, 2), "kaynak": _ad,
                        "oran": "~1/3",
                        "not": ("Son dilim değer çıpasına yükseltildi — fiyat "
                                "primi merdiveni tek başına kalsaydı bütün "
                                "değerleme çalışması plana hiç girmeyecekti.")}
        deger_cipali = True
    # ÇAPRAZ TEYİT: her hedefin ±%8 bandına düşen BAĞIMSIZ değer çıpaları.
    # Birden fazla bağımsız yöntem aynı bölgeyi gösteriyorsa hedef sağlamdır;
    # tek çıpalı hedef daha kırılgandır — kullanıcı bunu görerek karar verir.
    deger_adaylari = [(ad, v) for ad, v in adaylar
                      if not ad.startswith("fiyat") and "mevcut" not in ad]
    for h in hedefler:
        yakin = [ad for ad, v in deger_adaylari
                 if ad != h["kaynak"]
                 and abs(v - h["seviye"]) / h["seviye"] <= 0.08]
        h["teyit"] = ", ".join(dict.fromkeys(yakin)) if yakin else None
    if score >= 55:
        mod = "KÂR_ALMA_BÖLGESİ"
        mesaj = ("Fiyat pahalı bölgede — elde tutan için kademeli KÂR ALMA "
                 "buradan başlar. Tutmaya devam etmek de bir karardır ve "
                 "gerekçe ister.")
    elif deger_cipali:
        mod = "DEĞER_HEDEFLİ"
        mesaj = ("Satış seviyeleri değer çıpalarına bağlı: fiyat bu "
                 "bölgelere ulaştığında elindeki artık 'iyi şirket + iyi "
                 "fiyat' değil, 'iyi şirket + pahalı fiyat'tır — kademeli "
                 "kâr al.")
    else:
        mod = "FİYAT_PRİMLİ"
        mesaj = ("Değer çıpaları makul bantta değil/yok — seviyeler fiyat "
                 "primli merdivendir; asıl ağırlığı tez-bozan tetiklere "
                 "ver.")

    # 2) Pozisyon bağlamı (maliyet — çıpa DEĞİL)
    poz = None
    if maliyet and maliyet > 0:
        poz = {"maliyet": round(float(maliyet), 2),
               "guncel_kz": round(price / maliyet - 1, 3),
               "dilim_kazanclari": [round(h["seviye"] / maliyet - 1, 3)
                                    for h in hedefler],
               "uyari": ("Maliyet karar çıpası DEĞİLDİR — 'başa baş gelsin "
                         "de satayım' (disposition effect) klasik davranış "
                         "tuzağıdır. Satış kararı DEĞERE ve TEZE bağlanır, "
                         "giriş fiyatına değil.")}

    # 3) Tez-bozan tetikler (fiyattan bağımsız) — şirkete göre otomatik
    tet = ["Tez öncüllerinden biri kırılırsa (Claude raporundaki liste) "
           "fiyata bakmadan çık — en güçlü satış kuralı budur."]
    if (m.get("fcf_base") or 0) <= 0:
        tet.append("FCF bir dönem daha negatif kalırsa.")
    else:
        tet.append("FCF negatife dönerse.")
    nd = m.get("nd_ebitda")
    if nd is not None and nd > 0:
        esik = 3.5 if nd < 3.0 else round(nd + 1.0, 1)
        tet.append(f"Net Borç/EBITDA {esik}x üstüne çıkarsa "
                   f"(şu an {nd:.1f}x).")
    z = m.get("altman")
    if z is not None and not m.get("is_financial"):
        _abz = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                                ALTMAN_BOLGE["uretim"])
        if z < _abz["guvenli"]:
            tet.append(f"Altman-Z gri/kırmızı bölgede KALIRSA (şu an "
                       f"{z:.2f}; modelin güvenli eşiği "
                       f"{_abz['guvenli']}) — iyileşme görülmeli.")
    if (m.get("beneish") or {}).get("bolge"):
        tet.append("Beneish M-Skoru TEMİZ dışına (GRİ/RİSKLİ) kayarsa "
                   "(dikkat: modelin holdout yanlış-alarm oranı ~%17,5'tir "
                   "ve hızlı büyüyen / sermaye-yoğun şirketlerde MEŞRU "
                   "büyüme de eşiği aşabilir — Beneish 1999).")
    if any("Brüt marj eriyor" in t for _, t in (m.get("flags") or [])):
        tet.append("Brüt marj erimesi bir dönem daha sürerse.")
    tet.append("Analist revizyon yönü AŞAĞI'ya döner ve 30 günlük net "
               "revizyon ≤ −5 olursa.")
    _tr = ((m.get("stop") or {}).get("trailing") or {})
    if _tr.get("seviye"):
        tet.append(f"Chandelier trailing stop günlük KAPANIŞLA kırılırsa "
                   f"(şu an {_tr['seviye']:,.2f}; gap'te fiyat garantisi "
                   f"yok).")
    tet = tet[:7]

    # 4) Zaman stopu — katalizör haritasına bağlı
    gel = (m.get("catalysts") or {}).get("gelecek") or []
    if gel:
        e = gel[0]
        zs = (f"İlk kontrol noktası: {e['olay']} ({e['tarih']}, "
              f"{e['gun_kaldi']} gün). Bu olay tezi İLERLETMEZSE pozisyonu "
              f"yeniden değerlendir; iki katalizör üst üste 'boş' geçerse "
              f"zaman stopu uygula — ölü para da kayıptır.")
    else:
        zs = ("Takvimde kayıtlı katalizör yok: 90 günlük yeniden "
              "değerlendirme ritmi kur; tez ilerlemiyorsa zaman stopu "
              "uygula — ölü para da kayıptır.")

    # ── TUTMA UFKU ENTEGRASYONU: kısa ufukta zaman stopu TARİHLİDİR ve
    # planın ağırlığı hedeften TETİK+TAKVİME kayar (dürüst kenar yönetimi).
    uf = m.get("ufuk") or {}
    ufuk_notu = None
    if uf.get("kisa_ufuk"):
        zs = (f"UFUK STOPU ({uf.get('ufuk', '3-4 ay')}): son tarih "
              f"{uf.get('zaman_stopu_tarihi', '—')}. O gün geldiğinde "
              f"kâr/zarar ne olursa olsun pozisyon ya YENİDEN "
              f"gerekçelendirilir ya kapatılır — 'biraz daha bekleyeyim' "
              f"kısa ufkun sessiz maliyetidir. ")
        kz_ = uf.get("ufuk_katalizor") or []
        if kz_:
            zs += (f"Ufka düşen kritik olay: {kz_[0]['olay']} "
                   f"({kz_[0]['tarih']}) — kısa vadeli tezin asıl sınavı.")
        else:
            zs += ("Ufukta takvimli katalizör yok — tez neyle "
                   "gerçekleşecek? Cevap yoksa gerekçe zayıftır.")
        band_ = uf.get("tipik_bant_yuzde")
        if band_ and hedefler:
            h1 = hedefler[0]["seviye"] / price - 1
            if h1 > band_:
                ufuk_notu = (f"1. satış hedefi (+%{h1*100:.0f}) bu hissenin "
                             f"tipik {uf.get('ufuk')} bandının "
                             f"(±%{band_*100:.0f}) DIŞINDA — bu ufukta "
                             f"hedefe ulaşılamama olasılığı ciddidir; "
                             f"ağırlığı olay sonucuna ve zaman stopuna ver.")
            else:
                ufuk_notu = (f"1. satış hedefi (+%{h1*100:.0f}) tipik "
                             f"±%{band_*100:.0f} bandının İÇİNDE — "
                             f"ulaşılabilir bölgede; yine de garanti "
                             f"değildir.")

    return {"mod": mod, "mesaj": mesaj, "hedefler": hedefler,
            "ufuk_notu": ufuk_notu,
            "teyit_notu": ("'Çapraz teyit': o seviyenin ±%8 bandına düşen "
                           "bağımsız değer çıpaları (DCF, Monte Carlo, "
                           "analist). Çok çıpa = sağlam hedef; boş = tek "
                           "yönteme dayalı, temkinli oku."),
            "pozisyon": poz, "tetikler": tet, "zaman_stopu": zs,
            "iki_sistem_notu": (
                "DENETİM BULGUSU M: programda İKİ kâr-alma sistemi var ve "
                "farklı sorulara cevap veriyorlar — karıştırma. (1) BURADAKİ "
                "hedefler MEVCUT FİYATA ve değer çıpalarına dayanır; elde "
                "tutan için 'ne zaman kâr al' sorusudur. (2) Stop Tetiği "
                "bölümündeki 3R hedefi 1. ALIM DİLİMİNE ve stop mesafesine "
                "dayanır; yeni giriş yapan için 'risk-ödül nerede kapanır' "
                "sorusudur. Pozisyonun yoksa 3R'ye, varsa buradaki "
                "merdivene bak."),
            "kurallar": [
                "Her dilim ~1/3 — tek seferde tam çıkış da tam tutuş kadar "
                "duygusaldır.",
                "Skor 70+ (PAHALI/DUR) bölgesine tırmanırsa kalan pozisyonda "
                "tam çıkışı ciddiyetle değerlendir.",
                "Sattıktan sonra 'geri alım koşulu' yaz: hangi fiyat/koşulda "
                "geri girersin — yazmazsan FOMO seni tepeden geri sürükler."],
            "not": "Tavsiye değil — girişin aynası olan disiplin çerçevesi."}


def ufuk_analizi(m, hist, a):
    """
    TUTMA UFKU ANALİZİ — "daha da düşer mi?" sorusunun DÜRÜST ikamesi.
    Yönü kimse bilemez (bu araç dahil). Bilinebilir olanlar hesaplanır:
      1) TİPİK DALGALANMA BANDI: son 1 yılın gerçekleşen volatilitesi ufuk
         uzunluğuna ölçeklenir (±1σ) — "normal gürültü"nün sınırı. Fiyatın
         bu bant içinde gezinmesi TEZLE İLGİSİZDİR.
      2) PENCERE-İÇİ ÇEKİLME TARİHİ: son ~5 yılda ufuk uzunluğundaki her
         pencerede yaşanan en derin düşüşün medyanı ve kötü-durum (p90)
         değeri — "bu hisse bu sürede tipik olarak %kaç geri çekilir?"
      3) UFKA DÜŞEN KATALİZÖRLER: penceredeki bilanço/temettü olayları.
         Kısa ufukta sonucu çoğunlukla bunlar ve piyasa rejimi belirler.
      4) GİRİŞ DİLİMİ TUTARLILIĞI: dilim seviyeleri normal bandın
         içindeyse, oraya düşüş "plan bozulması" değil "planın işlemesi"dir.
      5) ZAMAN STOPU TARİHİ: ufuk sonunda pozisyon ya yeniden gerekçelenir
         ya kapatılır — süresiz "biraz daha bekleme" kapısı kapanır.
    Bu modül HAZIRLIK üretir, TAHMİN değil — çıktılarda açıkça yazar.
    """
    price = m.get("price")
    label = (a or {}).get("ufuk") or "6-12 ay"
    cfg = UFUKLAR.get(label) or UFUKLAR["6-12 ay"]
    out = {"ufuk": label, "kisa_ufuk": cfg["ay"][1] <= 4,
           "zaman_stopu_tarihi": str((pd.Timestamp.now()
                                      + pd.Timedelta(days=cfg["takvim_gunu"]))
                                     .date())}
    gel = (m.get("catalysts") or {}).get("gelecek") or []
    out["ufuk_katalizor"] = [e for e in gel
                             if 0 <= e.get("gun_kaldi", 9999)
                             <= cfg["takvim_gunu"]][:4]
    out["bilanco_sayisi"] = sum(1 for e in out["ufuk_katalizor"]
                                if "Kazanç" in e.get("olay", ""))
    try:
        close = (hist["Close"].dropna()
                 if (hist is not None and "Close" in hist) else None)
    except Exception:
        close = None
    if close is not None and len(close) >= 120 and price:
        r = np.log(close / close.shift(1)).dropna()
        sig_d = float(r.iloc[-252:].std()) if len(r) >= 60 else None
        if sig_d and sig_d > 0:
            sig_h = sig_d * math.sqrt(cfg["islem_gunu"])
            # ── DENETİM BULGUSU S: BANT ASİMETRİKTİR ──────────────────
            # Lognormal modelde exp(σ)−1 ile 1−exp(−σ) EŞİT DEĞİLDİR:
            # yukarı bant her zaman aşağı banttan geniştir (testte 16.3%
            # vs 14.0%). Tek sayı "±%X" diye sunulunca AŞAĞI hareketler
            # olduğundan daha "normal" görünüyordu — ve dilim tutarlılık
            # kontrolü (bir DÜŞÜŞ karşılaştırması) yukarı bandı kullandığı
            # için düşüşleri sistematik olarak hoş görüyordu.
            # DÜZELTME: iki yön ayrı raporlanır; düşüş karşılaştırmaları
            # AŞAĞI bandı kullanır.
            out["tipik_bant_yuzde"] = round(math.exp(sig_h) - 1, 3)
            out["bant_yukari_yuzde"] = round(math.exp(sig_h) - 1, 3)
            out["bant_asagi_yuzde"] = round(1 - math.exp(-sig_h), 3)
            out["bant_alt"] = round(price * math.exp(-sig_h), 2)
            out["bant_ust"] = round(price * math.exp(sig_h), 2)
            out["bant_notu"] = (
                f"Bant SİMETRİK DEĞİLDİR (lognormal): yukarı "
                f"+%{(math.exp(sig_h)-1)*100:.1f}, aşağı "
                f"−%{(1-math.exp(-sig_h))*100:.1f}. Düşüş "
                f"değerlendirmelerinde AŞAĞI bandı esastır.")
        W = cfg["islem_gunu"]
        if len(close) >= W + 40:
            arr = close.to_numpy(dtype=float)
            dips = []
            for s in range(0, len(arr) - W, 5):
                seg = arr[s:s + W]
                run_max = np.maximum.accumulate(seg)
                dips.append(float((seg / run_max - 1.0).min()))
            if dips:
                out["pencere_cekilme_medyan"] = round(
                    float(np.median(dips)), 3)
                out["pencere_cekilme_kotu"] = round(
                    float(np.percentile(dips, 10)), 3)
                out["pencere_sayisi"] = len(dips)
    kd = m.get("kademe") or {}
    dil = kd.get("dilimler") or []
    # BULGU S: dilim kontrolü bir DÜŞÜŞ karşılaştırmasıdır → AŞAĞI bant.
    band = out.get("bant_asagi_yuzde") or out.get("tipik_bant_yuzde")
    if dil and band and price:
        d_son = (dil[-1] or {}).get("seviye")
        if d_son:
            dusus = 1 - d_son / price
            out["dilim_tutarlilik"] = (
                f"Son giriş dilimi fiyatın %{dusus*100:.0f} altında; bu "
                f"hissenin tipik {label} AŞAĞI bandı −%{band*100:.0f} "
                f"(yukarı bant daha geniştir: "
                f"+%{(out.get('bant_yukari_yuzde') or band)*100:.0f}). "
                + ("Dilimlere iniş NORMAL dalgalanmanın içindedir — oraya "
                   "düşüş tez kırılması değil planın çalışmasıdır (tez-bozan "
                   "tetikler ayrıca izlenir)." if dusus <= band * 1.1 else
                   "Dilimler tipik bandın DIŞINDA — oraya iniş sıradan "
                   "gürültüyle açıklanamaz; gelirse önce 'neden?' sorusu ve "
                   "tetik listesi devreye girer."))
    if out["kisa_ufuk"]:
        out["durustluk"] = (
            "KISA UFUK GERÇEĞİ: Temel analizin kenarı çeyrekler-yıllar "
            "ölçeğinde gerçekleşir; 3-4 aylık pencerede fiyatı çoğunlukla "
            "ufuktaki bilanço ve genel piyasa havası belirler. Bu modül YÖN "
            "tahmini YAPMAZ — 'daha düşer mi?' sorusunun cevabı kimsede "
            "yoktur. Verdiği şey: tipik aralık, olay takvimi ve karar "
            "kuralları. Adil değere yürüyüş bu pencerede tamamlanmayabilir; "
            "kısa ufukta plan, hedeften çok TETİK ve ZAMAN STOPUNA yaslanır.")
    else:
        out["durustluk"] = (
            "Bu modül yön tahmini yapmaz; tipik dalgalanma aralığı, olay "
            "takvimi ve zaman-stopu disiplini üretir. Değer tezinin "
            "gerçekleşmesi tipik olarak çeyrekler-yıllar alır.")
    return out


def veri_tazeligi(m, b):
    """
    BİLANÇO GÜNÜ PENCERESİ DEDEKTÖRÜ (rapor bulgusu): SEC'e yeni 10-Q/10-K
    düşer düşmez EDGAR API'sinde görünür (<1 dk), ama Yahoo tabloları
    günlerce eski dönemi gösterebilir. Yaklaşık kontrol: dosyalama tarihi −
    tipik gecikme (10-Q ~45g, 10-K ~75g) Yahoo'nun en yeni dönem sonundan
    belirgin yeniyse uyarı düşer. Fiyatın ~15 dk gecikmeli olabileceği de
    açıkça not edilir — gün içi tetikler gerçek zamanlı DEĞİLDİR.
    """
    out = {"fiyat_notu": "Yahoo fiyatı ABD borsalarında ~15 dk gecikmeli "
                         "olabilir — gün içi stop/alım tetikleri gerçek "
                         "zamanlı DEĞİLDİR."}
    fl = b.get("filings") or []
    adaylar = []
    for f in fl:
        try:
            if f.get("form") in ("10-K", "10-Q") and f.get("tarih"):
                adaylar.append((f["form"], pd.Timestamp(f["tarih"])))
        except Exception:
            continue
    if not adaylar:
        return out
    form, fdate = max(adaylar, key=lambda x: x[1])
    out["son_dosyalama"] = {"form": form, "tarih": str(fdate.date())}
    donemler = []
    for df_ in (b.get("inc"), b.get("qcf")):
        try:
            if df_ is not None:
                donemler += [pd.Timestamp(c) for c in df_.columns]
        except Exception:
            pass
    donemler = [d for d in donemler if pd.notna(d)]
    if not donemler:
        return out
    son_donem = max(donemler)
    out["yahoo_son_donem"] = str(son_donem.date())
    # B8-9: SEC son teslim tarihleri — Büyük hızlandırılmış: 10-K 60 /
    # 10-Q 40 · Hızlandırılmış: 75 / 40 · Hızlandırılmamış: 90 / 45.
    # Kod her şirkete 10-Q için 45 veriyordu = HIZLANDIRILMAMIŞ değeri;
    # çoğu büyük hissede gerçek sınır 40 gündür (5 gün fazla tolerans).
    _mc_ = _num(m.get("mcap")) or 0
    if _mc_ >= 700e6:
        tipik = 60 if form == "10-K" else 40
    elif _mc_ >= 75e6:    # accelerated ara katman (yaklaşık; ölçüt float)
        tipik = 75 if form == "10-K" else 40
    else:
        tipik = 90 if form == "10-K" else 45
    beklenen = fdate - pd.Timedelta(days=tipik)
    if son_donem < beklenen - pd.Timedelta(days=20):
        gap = int((beklenen - son_donem).days)
        out["uyari"] = (f"SEC'e {form} dosyalandı ({fdate.date()}) ama "
                        f"Yahoo tablolarının en yeni dönemi "
                        f"{son_donem.date()} — Yahoo ~{gap} gün geride "
                        f"görünüyor. SEC-birincil kalemler günceldir; "
                        f"bilanço detayını 10-Q/K'dan doğrula. "
                        f"[yaklaşık kontrol]")
    return out


def parse_short(info):
    """Açığa satış (FINRA, ~2 hafta gecikmeli)."""
    spf = _num(info.get("shortPercentOfFloat"))
    sr = _num(info.get("shortRatio"))
    ss = _num(info.get("sharesShort"))
    sp = _num(info.get("sharesShortPriorMonth"))
    if spf is None and sr is None and ss is None:
        return None
    tarih = info.get("dateShortInterest")
    try:
        tarih = str(pd.Timestamp(tarih, unit="s").date()) if tarih else None
    except Exception:
        tarih = None
    return {"float_yuzdesi": spf, "kapatma_gunu": sr,
            "kisa_pozisyon_adet": ss, "onceki_ay_adet": sp,
            "aylik_degisim": (round(ss / sp - 1, 3) if (ss and sp) else None),
            "veri_tarihi": tarih,
            "not": "FINRA verisi ~2 hafta gecikmelidir."}


def parse_holders(major, inst):
    """Kurumsal sahiplik: toplam yüzdeler + en büyük 5 kurum (13F özeti)."""
    out, top = {}, []
    try:
        if major is not None and not getattr(major, "empty", True):
            d = major.copy()
            if d.shape[1] == 1:
                ser = d.iloc[:, 0]
                ser.index = [str(i) for i in d.index]
            else:
                ser = pd.Series(dict(zip(d.iloc[:, 1].astype(str),
                                         d.iloc[:, 0])))

            def pick(*keys):
                for k in ser.index:
                    if any(q in str(k).lower() for q in keys):
                        return _num(ser[k])
                return None
            out["kurum_yuzde"] = pick("institutionspercent",
                                      "institutions %")
            out["insider_yuzde"] = pick("insiderspercent", "insider")
            out["kurum_sayisi"] = pick("institutionscount",
                                       "number of institutions")
    except Exception:
        pass
    try:
        if inst is not None and not getattr(inst, "empty", True):
            d = inst.copy()
            d.columns = [str(c) for c in d.columns]
            col = {c.lower(): c for c in d.columns}
            c_h = col.get("holder")
            c_p = col.get("pctheld") or col.get("% out")
            c_c = col.get("pctchange")
            for _, r in d.head(5).iterrows():
                top.append({"kurum": (str(r.get(c_h, "")).strip()[:40]
                                      if c_h else ""),
                            "pay_yuzde": _num(r.get(c_p)) if c_p else None,
                            "ceyreklik_degisim": (_num(r.get(c_c))
                                                  if c_c else None)})
    except Exception:
        pass
    if not out and not top:
        return None
    out["en_buyuk_5"] = top or None
    out["not"] = ("13F kaynaklı; ~45 gün gecikmeli — momentum değil, trend "
                  "sinyali.")
    return out


def parse_ipo(info, hist):
    """Genç halka arz tespiti + tahminî lock-up (180 gün örfü)."""
    ts, ep = None, (info.get("firstTradeDateEpochUtc")
                    or info.get("firstTradeDateMilliseconds"))
    try:
        if ep:
            ts = pd.Timestamp(ep, unit="ms" if ep > 1e11 else "s")
    except Exception:
        ts = None
    if ts is None and hist is not None and len(hist):
        try:
            ts = pd.Timestamp(hist.index[0])
        except Exception:
            return None
    if ts is None:
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    yas = int((pd.Timestamp.now() - ts).days)
    if yas > 600:
        return {"durum": "ESKI_SIRKET", "ipo_tarihi": str(ts.date()),
                "yas_gun": yas}
    lock = ts + pd.Timedelta(days=180)
    kaldi = int((lock - pd.Timestamp.now()).days)
    return {"durum": "LOCKUP_GELECEK" if kaldi > 0 else "LOCKUP_DOLDU",
            "ipo_tarihi": str(ts.date()), "yas_gun": yas,
            "lockup_tahmini": str(lock.date()), "kalan_gun": kaldi,
            "not": "180 gün örfî varsayım [TÜRETİLMİŞ]; gerçek şart 424B4 "
                   "prospektüsündedir — Claude webden doğrular."}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peers(tickers: tuple) -> list:
    """Kullanıcının girdiği rakip ticker'ları için hafif çarpan seti çek."""
    import yfinance as yf
    rows = []
    for t in tickers[:6]:
        try:
            info = {}
            tkp = yf.Ticker(t, session=_yf_session())
            for getter in ("get_info", "info"):
                try:
                    obj = getattr(tkp, getter)
                    info = obj() if callable(obj) else obj
                    if info:
                        break
                except Exception:
                    info = {}
            if not info:
                continue
            rows.append({"ticker": t,
                         "isim": (info.get("shortName")
                                  or info.get("longName") or t)[:24],
                         "pe_f": _num(info.get("forwardPE")),
                         "pe_t": _num(info.get("trailingPE")),
                         "ev_ebitda": _num(info.get("enterpriseToEbitda")),
                         "ps": _num(info.get("priceToSalesTrailing12Months")),
                         "brut_marj": _num(info.get("grossMargins")),
                         "gelir_buyume": _num(info.get("revenueGrowth")),
                         "mcap": _num(info.get("marketCap"))})
        except Exception:
            continue
    return rows


def build_peer_table(m, rows):
    """Kendi hissemiz + rakipler: tablo, medyan ve göreli teşhis notları."""
    if not rows:
        return None
    me = {"ticker": m["ticker"] + " ★", "isim": (m.get("name") or "")[:24],
          "pe_f": m.get("pe_f"), "pe_t": m.get("pe_t"),
          "ev_ebitda": m.get("ev_ebitda"), "ps": m.get("ps"),
          "brut_marj": latest(m.get("gm") or {}),
          "gelir_buyume": m.get("rev_cagr3"), "mcap": m.get("mcap")}

    def med(key):
        v = sorted(x[key] for x in rows if x.get(key) is not None)
        return _medyan(v)                     # BULGU U
    medyan = {k: med(k) for k in ("pe_f", "pe_t", "ev_ebitda", "ps",
                                  "brut_marj", "gelir_buyume")}
    goreli = []
    for key, ad, iyi_yuksek in (("pe_f", "Forward F/K", False),
                                ("ev_ebitda", "EV/EBITDA", False),
                                ("ps", "F/S", False),
                                ("brut_marj", "Brüt marj", True),
                                ("gelir_buyume", "Gelir büyümesi", True)):
        my, md = me.get(key), medyan.get(key)
        if my is None or not md:
            continue
        oran = my / md
        if not iyi_yuksek and oran >= 1.3:
            goreli.append(f"{ad} rakip medyanının {oran:.1f}x ÜSTÜNDE — "
                          f"prim ödeniyor; primin gerekçesi sorgulanmalı.")
        elif not iyi_yuksek and oran <= 0.7:
            goreli.append(f"{ad} rakip medyanının {oran:.1f} katı — iskonto "
                          f"var; 'neden ucuz?' sorusu açık.")
        elif iyi_yuksek and oran >= 1.2:
            goreli.append(f"{ad} medyanın {oran:.1f}x üstünde — operasyonel "
                          f"üstünlük sinyali.")
        elif iyi_yuksek and oran <= 0.8:
            goreli.append(f"{ad} medyanın {oran:.1f} katı — rakiplerin "
                          f"gerisinde.")
    return {"tablo": [me] + rows, "medyan": medyan, "goreli": goreli or None,
            "not": "Kaynak Yahoo; rakip listesi kullanıcı seçimidir."}


def resolve_peg(info, pe_t):
    """PEG için savunmacı zincir: Yahoo alanları → kendi hesabımız."""
    for key in ("trailingPegRatio", "pegRatio"):
        v = _num(info.get(key))
        if v and 0 < v < 50:
            return v, f"yahoo:{key}"
    g = _num(info.get("earningsGrowth"))
    if pe_t and g and g > 0.01:
        return pe_t / (g * 100), "hesap:PE/earningsGrowth"
    return None, None


def suggest_assumptions(b):
    """
    Şirketin KENDİ verisinden makul DCF varsayımları türet.
    Amaç: kullanıcının %8 gibi jenerik bir sayıyla uğraşmaması. Her öneri
    bir GEREKÇEYLE döner; kullanıcı isterse override eder.

    Mantık:
      - Büyüme: gelir 3Y CAGR ile analist beklentisinin HARMANLA, sonra
        sürdürülebilirlik için törpüle (hiçbir şirket sonsuza dek >%25 büyümez).
      - İskonto: büyük/kârlı/net-nakit şirkete düşük, küçük/borçlu/zararlı
        şirkete yüksek risk primi.
      - Terminal: gelişmiş ekonomi nominal büyümesi ~%2.5 sabit.
      - Yıl: hızlı büyüyene daha uzun projeksiyon (rekabet avantajı süresi).
    """
    info = b.get("info", {})
    inc, cf = b.get("inc"), b.get("cf")

    rev = by_year(get_row(inc, ROWS["revenue"]))
    rev_cagr = cagr(rev, 3)
    analyst_g = _num(info.get("earningsGrowth"))          # yıllık, fraksiyon
    rev_growth_info = _num(info.get("revenueGrowth"))

    # --- büyüme sinyallerini harmanla -------------------------------------
    signals, weights, why = [], [], []
    if rev_cagr is not None:
        signals.append(rev_cagr); weights.append(0.5)
        why.append(f"3Y gelir CAGR %{rev_cagr*100:.0f}")
    if rev_growth_info is not None and -0.5 < rev_growth_info < 1.5:
        signals.append(rev_growth_info); weights.append(0.3)
        why.append(f"son yıl gelir büyümesi %{rev_growth_info*100:.0f}")
    if analyst_g is not None and -0.5 < analyst_g < 1.5:
        signals.append(analyst_g); weights.append(0.2)
        why.append(f"analist kâr büyümesi beklentisi %{analyst_g*100:.0f}")

    if signals:
        raw_g = sum(s*w for s, w in zip(signals, weights)) / sum(weights)
    else:
        raw_g = 0.08
        why.append("veri yok, jenerik %8 tabanı")

    # Sürdürülebilirlik törpüsü: yüksek büyümeyi kademeli kırp (5 yıllık
    # projeksiyonda hiçbir şirket ilk yılki tempoyu sonuna kadar koruyamaz).
    if raw_g > 0.25:
        growth = 0.25 + (raw_g - 0.25) * 0.4     # %25 üstünü %40 katsayıyla
        why.append("→ sürdürülebilirlik için %25 üstü törpülendi")
    elif raw_g < 0:
        growth = max(raw_g, -0.05)               # negatifi -%5'te sınırla
    else:
        growth = raw_g
    growth = max(-0.05, min(growth, 0.35))

    # --- iskonto (risk primi) ---------------------------------------------
    mcap = _num(info.get("marketCap")) or 0
    total_debt = _num(info.get("totalDebt")) or 0
    cash = _num(info.get("totalCash")) or 0
    ni = by_year(get_row(inc, ROWS["net_income"]))
    profitable = (latest(ni) or 0) > 0
    fcf = by_year(get_row(cf, ROWS["fcf"]))
    fcf_pos = (latest(fcf) or 0) > 0

    # CAPM (Damodaran/CFA): Re = Rf + β × ERP — β varsa bilimsel yol budur.
    # Uç β'lar tahmin gürültüsü taşır → [0.5, 2.2] bandına kırpılır.
    # β yoksa eski sezgisel taban devrede kalır (etiketiyle).
    beta = _num(info.get("beta"))
    net_debt = total_debt - cash
    # (a2) Kenar çubuğundaki canlı Rf/ERP girdisi; yoksa koddaki sabit.
    rf, erp, _rf_kaynak = rf_erp_guncel()
    if beta is not None and 0.05 < beta < 4.0:
        # B4-5: KIRPMA yerine Blume (1971) shrinkage: 0.67×ham + 0.33×1.
        # Kırpma kaba bir alettir — aralığın içindeki gürültülü betalara
        # hiç dokunmaz. Betalar 1'e geri döner; Bloomberg/Value Line
        # standardı budur ve ham OLS betasından daha iyi ÖNGÖRÜ verir.
        b_ = 0.67 * beta + 0.33 * 1.0
        disc = rf + b_ * erp
        disc_why = [f"CAPM: Rf %{rf*100:.2f} ({_rf_kaynak}) + β {b_:.2f} "
                    f"(Blume düzeltmeli, ham {beta:.2f}) × "
                    f"ERP %{erp*100:.2f}"]
    else:
        disc = 0.09
        disc_why = ["β verisi yok → sezgisel taban %9"]
        if mcap > 200e9:
            disc -= 0.01; disc_why.append("mega-cap (-1)")
        elif mcap < 10e9:
            disc += 0.01; disc_why.append("orta/küçük ölçek (+1)")
    # CAPM'in kapsamadığı ek risk primleri (boyut/sıkıntı primi geleneği):
    # B4-6: KÜÇÜK ŞİRKET PRİMİ KALDIRILDI. Damodaran açıkça: "I have
    # never used a small cap premium." Boyut etkisi 1980'lerden sonra
    # kayboldu (Cochrane, Ang). Mekanik +%1.5 prim, küçük kaliteli
    # şirketleri sistematik olarak pahalı gösteriyordu (ölçüm: iskonto
    # %10.33 → %8.69, adil değer %+20).
    if mcap and net_debt > mcap * 0.5:
        disc += 0.010; disc_why.append("yüksek kaldıraç (+1)")
    if not profitable:
        disc += 0.015; disc_why.append("zarar ediyor (+1.5)")
    if not fcf_pos:
        disc += 0.010; disc_why.append("FCF negatif (+1)")
    disc = max(0.07, min(disc, 0.18))

    # --- projeksiyon yılı --------------------------------------------------
    years = 7 if growth > 0.15 else (6 if growth > 0.08 else 5)

    return {
        "growth": round(growth, 4),
        "discount": round(disc, 4),
        "terminal": 0.025,
        "years": years,
        "sbc_adjust": True,
        "net_debt_adjust": True,
        "_why_growth": "; ".join(why),
        "_why_discount": ", ".join(disc_why),
        "_raw_growth": raw_g,
    }



# ---------------------------------------------------------------------------
# 4) KALİTE SKORLARI — PIOTROSKI F, ALTMAN Z
# ---------------------------------------------------------------------------

def piotroski_f(ni, cfo, ta, ltd, ca, cl, sh, gp, rev):
    """
    9 kriterlik Piotroski F. Veri eksik kriterler atlanır → skor 'x / mevcut'.
    Girdiler {yıl→değer} sözlükleri.
    """
    checks, passed = [], []

    def add(name, cond):
        if cond is None:
            return
        checks.append(name)
        if cond:
            passed.append(name)

    def ratio(a, b, k):
        x, y = latest(a, k), latest(b, k)
        return (x / y) if (x is not None and y not in (None, 0)) else None

    # ── B5-3 ─────────────────────────────────────────────
    # Piotroski (2000) ROA ve varlık devrini DÖNEM BAŞI (t−1)
    # toplam varlığa böler; kod dönem SONU kullanıyordu.
    # ÖLÇÜM: varlık +%40 büyüyen şirket dönem sonu bazla ΔROA
    # ve Δvarlık devrini BİRDEN kaybediyor (0/2), dönem
    # başıyla ikisini de alıyor (2/2) — 9 üzerinden 2 puan,
    # 'güçlü' eşiğini (7) tek başına çevirir.
    def oran_bas(pay, payda, k=0):
        x, y = latest(pay, k), latest(payda, k + 1)
        return (x / y) if (x is not None and y not in (None, 0)) else None

    roa0, roa1 = oran_bas(ni, ta, 0), oran_bas(ni, ta, 1)
    add("ROA > 0", None if roa0 is None else roa0 > 0)
    add("CFO > 0", None if latest(cfo) is None else latest(cfo) > 0)
    add("ΔROA > 0", None if None in (roa0, roa1) else roa0 > roa1)
    ni0 = latest(ni)
    add("CFO > Net Kâr (tahakkuk)", None if None in (latest(cfo), ni0)
        else latest(cfo) > ni0)
    def _kald(k=0):                     # B5-3: kaldıraç ORTALAMA varlığa
        d_, t0, t1 = latest(ltd, k), latest(ta, k), latest(ta, k + 1)
        if d_ is None or t0 is None:
            return None
        ort = ((t0 + t1) / 2) if t1 else t0
        return (d_ / ort) if ort else None

    lev0, lev1 = _kald(0), _kald(1)
    add("Kaldıraç ↓", None if None in (lev0, lev1) else lev0 <= lev1)
    cr0, cr1 = ratio(ca, cl, 0), ratio(ca, cl, 1)
    add("Cari oran ↑", None if None in (cr0, cr1) else cr0 > cr1)
    s0, s1 = latest(sh, 0), latest(sh, 1)
    add("Hisse sayısı artmadı", None if None in (s0, s1) else s0 <= s1 * 1.005)
    gm0, gm1 = ratio(gp, rev, 0), ratio(gp, rev, 1)
    add("Brüt marj ↑", None if None in (gm0, gm1) else gm0 > gm1)
    at0, at1 = oran_bas(rev, ta, 0), oran_bas(rev, ta, 1)
    add("Varlık devri ↑", None if None in (at0, at1) else at0 > at1)

    return {"score": len(passed), "max": len(checks), "passed": passed}


ALTMAN_BOLGE = {
    "uretim": {"tehlike": 1.81, "guvenli": 2.99},
    "hizmet": {"tehlike": 1.10, "guvenli": 2.60},   # Z'' eşikleri
    # Defter özsermayesi negatif olduğunda (geri alım kaynaklı) piyasa
    # değerine düşülür; eşikler aynı kalır ama etiket farklıdır.
    "hizmet_piyasa_ikamesi": {"tehlike": 1.10, "guvenli": 2.60},
}


def altman_z(wc, re_, ebit, mcap, tl, rev, ta, model="uretim", bve=None):
    """
    Altman Z — SEKTÖRE GÖRE MODEL (denetim düzeltmesi).
    Orijinal Z (1.2/1.4/3.3/0.6/1.0) yalnızca halka açık ÜRETİM şirketleri
    için kalibre edildi (Altman 1968, 66 üretici firma). Teknoloji/hizmet
    gibi varlık-hafif şirketlerde satış/varlık terimi (X5) yanıltıcıdır —
    bu şirketlerde Z''(non-manufacturing) kullanılır:
        Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4   (X5 YOK)
        güvenli > 2.60 · gri 1.10–2.60 · sıkıntı < 1.10
    Finansal sektörde her iki model de ANLAMSIZ → çağrı yapılmaz.
    X4 PAYDASI (bütünsel rapor G1 doğrulaması): her iki modelde de payda
    TOPLAM YÜKÜMLÜLÜKLERDİR (tl = ROWS['total_liab'] = "Total Liabilities
    Net Minority Interest") — yalnız faizli "toplam borç" DEĞİL. Bu dosya
    zaten kanonikti; gerileme olmasın diye kalıcı not olarak sabitlendi.
    Dönen: (skor, model_adı) — bölge eşikleri ALTMAN_BOLGE'den okunur.
    """
    vals = [_num(v) for v in (wc, re_, ebit, mcap, tl, ta)]
    if any(v is None for v in vals) or vals[-1] == 0 or vals[4] == 0:
        return None, model
    wc, re_, ebit, mcap, tl, ta = vals
    x1, x2, x3 = wc / ta, re_ / ta, ebit / ta
    if model == "hizmet":
        # ── DENETİM BULGUSU P (FORMÜL HATASI) ─────────────────────────
        # Z'' (Altman'ın üretim-dışı / gelişen piyasa modeli) X4'te
        # DEFTER özsermayesi kullanır, PİYASA değeri değil. Üç varyant:
        #   Z   (halka açık üretim) : X4 = piyasa değeri / toplam borç
        #   Z'  (özel şirket)       : X4 = DEFTER değeri / toplam borç
        #   Z'' (üretim-dışı)       : X4 = DEFTER değeri / toplam borç
        # Kod her iki modelde de mcap kullanıyordu. Z'' tam olarak
        # teknoloji/hizmet şirketlerine uygulandığı — yani piyasa/defter
        # oranı en yüksek olanlara — hata SİSTEMATİK OLARAK "daha güvenli"
        # yönde çalışıyordu; iflas mesafesi olduğundan büyük görünüyordu.
        _bv = _num(bve)
        if _bv is not None and _bv > 0:
            return (6.56 * x1 + 3.26 * x2 + 6.72 * x3
                    + 1.05 * (_bv / tl)), "hizmet"
        # Defter özsermayesi negatif/yok: modern geri-alım şirketlerinde
        # (MCD, SBUX, HD gibi) özkaynak eksiye döner ve kanonik Z'' onları
        # yanlışlıkla "iflas" gösterir. Bu durumda piyasa değerine düşülür
        # ve model adı bunu AÇIKÇA söyler — sessiz sapma bırakılmaz.
        return (6.56 * x1 + 3.26 * x2 + 6.72 * x3
                + 1.05 * (mcap / tl)), "hizmet_piyasa_ikamesi"
    rev_ = _num(rev)
    if rev_ is None:
        return None, model
    # Z (1968, halka açık üretim): X4 kanonik olarak PİYASA değeridir — burası
    # doğruydu, değiştirilmedi.
    return (1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * (mcap / tl)
            + 1.0 * rev_ / ta), "uretim"


# ---------------------------------------------------------------------------
# 5) METRİK MOTORU — tüm hattı tek geçişte kur
# ---------------------------------------------------------------------------


def plan_saglamasi(m):
    """
    ALIM SEVİYELERİNİN ÖZ-DENETİMİ (bu turda eklendi).

    Neden var: bu denetimde bulunan hataların HEPSİ 'sessiz yanlış sayı'
    sınıfındaydı — merdiven ters dönüyordu, stop %63 aşağıdaydı, tek
    dokunuş 4 sayılıyordu. Hiçbiri çökme üretmedi, hepsi makul GÖRÜNEN
    sayı üretti. Bu fonksiyon o sınıfa karşı kalıcı bir bariyerdir:
    ekrana çıkan her seviye önce burada sınanır.

    Dönen: sorun listesi (boşsa plan tutarlı). Sorun varsa arayüz planı
    GÖSTERMEZ; uydurma seviye vermektense hiç vermemek doğrudur.
    """
    s = []
    kd, sp = m.get("kademe") or {}, m.get("stop") or {}
    price, atr = m.get("price"), m.get("atr22")
    if not price or price <= 0:
        return ["Fiyat yok/geçersiz — seviye üretilemez."]

    # ── 1) DİLİMLER ───────────────────────────────────────────────────
    dl = kd.get("dilimler") or []
    sev = [d.get("seviye") for d in dl]
    if dl:
        if any(x is None or x <= 0 for x in sev):
            s.append("Dilim seviyelerinden biri boş ya da sıfır/negatif.")
        else:
            if any(sev[i] <= sev[i + 1] for i in range(len(sev) - 1)):
                s.append(f"Dilim merdiveni azalan DEĞİL: {sev} — aşağı inen "
                         f"her dilim bir öncekinden düşük olmalı.")
            if sev[0] > price * 1.005:
                s.append(f"1. dilim ({sev[0]:,.2f}) fiyatın ÜSTÜNDE "
                         f"({price:,.2f}) — kademeli giriş iskonto ister.")
            if sev[-1] < price * 0.45:
                s.append(f"Son dilim ({sev[-1]:,.2f}) fiyatın %55'inden "
                         f"aşağıda — bu ufukta ulaşılamaz.")
        # yapısal etiketli dilim GERÇEKTEN bir bölgede mi
        bl = (m.get("sr") or {}).get("destekler") or []
        for d in dl:
            if d.get("cipa") == "destek bölgesi" and d.get("seviye"):
                if not any(z["bant_alt"] * 0.99 <= d["seviye"]
                           <= z["bant_ust"] * 1.01 for z in bl):
                    s.append(f"{d.get('dilim', '?')}. dilim 'destek bölgesi' "
                             f"etiketli ama {d['seviye']:,.2f} hiçbir "
                             f"raporlanan bölgenin içinde değil.")

    # ── 2) DESTEK/DİRENÇ BÖLGELERİ ───────────────────────────────────
    sr = m.get("sr") or {}
    for ad, ters in (("destekler", False), ("direncler", True)):
        for z in (sr.get(ad) or []):
            if z["bant_alt"] > z["bant_ust"]:
                s.append(f"{ad}: bozuk bant ({z['bant_alt']}>{z['bant_ust']}).")
            if not ters and z["bant_ust"] > price * 1.005:
                s.append(f"Destek bölgesi {z['bant_ust']:,.2f} fiyatın "
                         f"ÜSTÜNDE — destek tanımına aykırı.")
            if ters and z["bant_alt"] < price * 0.995:
                s.append(f"Direnç bölgesi {z['bant_alt']:,.2f} fiyatın "
                         f"ALTINDA — direnç tanımına aykırı.")
            if (z.get("dokunma") or 0) < 1:
                s.append(f"{ad}: dokunma sayısı 0 olan bölge raporlanmış.")

    # ── 3) STOP ve HEDEF ─────────────────────────────────────────────
    gs = sp.get("giris_stop") or {}
    if gs.get("seviye") is not None and dl and sev and sev[0]:
        st = gs["seviye"]
        if st <= 0:
            s.append(f"Giriş stopu sıfır/negatif ({st:,.2f}).")
        elif st >= sev[0]:
            s.append(f"Giriş stopu ({st:,.2f}) 1. dilimin ({sev[0]:,.2f}) "
                     f"ÜSTÜNDE — anında tetiklenirdi.")
        elif atr and atr > 0:
            kat = (sev[0] - st) / atr
            if kat > 8:
                s.append(f"Stop mesafesi {kat:.1f}×ATR — aşırı geniş; "
                         f"pozisyon boyutu anlamsız küçülür ve stop fiilen "
                         f"koruma sağlamaz.")
            elif kat < 1.0:
                s.append(f"Stop mesafesi {kat:.1f}×ATR — aşırı dar; normal "
                         f"gürültüde tetiklenir.")
        rh = sp.get("r_hedef") or {}
        if rh.get("seviye") is not None:
            if rh["seviye"] <= sev[0]:
                s.append(f"Kâr hedefi ({rh['seviye']:,.2f}) girişin ALTINDA.")
            elif st < sev[0]:
                kat_r = (rh["seviye"] - sev[0]) / (sev[0] - st)
                _kat_h = _num(rh.get("kat")) or 3.0
                if abs(kat_r - _kat_h) > 0.05:
                    s.append(f"R hedefi {_kat_h:.0f}R değil ({kat_r:.2f}R) — "
                             f"stop/hedef matematiği tutarsız.")

    # ── 4) POZİSYON ──────────────────────────────────────────────────
    ph = ((sp.get("pozisyon") or {}).get("hesap") or {})
    # K3 DÜZELTMESİ: hesap sözlüğü "portfoy" anahtarını HİÇ üretmiyor
    # (yalnız portfoy_pct var) — eski koşul daima False'tu ve %60
    # yoğunlaşma bekçisi ölü koddu.
    pay = ph.get("portfoy_pct")
    if pay is not None and pay > 0.60:
        s.append(f"Önerilen pozisyon portföyün %{pay*100:.0f}'i — "
                 f"tek hissede aşırı yoğunlaşma.")

    # ── 5) PAHALI DAL: izleme eşiği ──────────────────────────────────
    if kd.get("mod") == "PLAN_YOK_IZLE":
        e = kd.get("izleme_esigi")
        if e is None or e <= 0:
            s.append("İzleme eşiği boş/geçersiz.")
        elif e >= price:
            s.append(f"İzleme eşiği ({e:,.2f}) fiyatın ({price:,.2f}) "
                     f"üstünde/eşit — 'bekle' demek anlamsız olur.")
        elif e < price * 0.60:
            s.append(f"İzleme eşiği ({e:,.2f}) fiyatın %40'ından fazla "
                     f"altında — plan değil duvar.")
    return s

def compute_metrics(b: dict, a: dict) -> dict:
    """b: fetch_bundle çıktısı, a: varsayımlar (fraksiyon). Saf fonksiyon."""
    info, inc, bal, cf, qcf = b["info"], b["inc"], b["bal"], b["cf"], b["qcf"]
    m = {"ticker": b["ticker"], "fetched_at": b["fetched_at"]}

    m["name"] = info.get("longName") or info.get("shortName") or b["ticker"]
    m["sector"] = info.get("sector") or "—"
    m["industry"] = info.get("industry") or "—"
    m["is_ozeti"] = info.get("longBusinessSummary")
    m["ccy"] = info.get("currency") or "USD"
    m["fin_ccy"] = info.get("financialCurrency") or m["ccy"]
    m["ccy_mismatch"] = (m["ccy"] != m["fin_ccy"])
    cur = "$" if m["ccy"] == "USD" else m["ccy"] + " "
    m["cur"] = cur

    m["price"] = _num(info.get("currentPrice")) or b["fast"]["price"]

    # --- yıllık seriler ---------------------------------------------------
    rev = by_year(get_row(inc, ROWS["revenue"]))
    gp = by_year(get_row(inc, ROWS["gross"]))
    oi = by_year(get_row(inc, ROWS["op_income"]))
    ni = by_year(get_row(inc, ROWS["net_income"]))
    ebitda_s = by_year(get_row(inc, ROWS["ebitda"]))
    cfo = by_year(get_row(cf, ROWS["cfo"]))
    capex = by_year(get_row(cf, ROWS["capex"]))
    sbc = by_year(get_row(cf, ROWS["sbc"]))
    fcf = by_year(get_row(cf, ROWS["fcf"]))
    if not fcf and cfo and capex:
        fcf = {y: cfo[y] - abs(capex[y]) for y in cfo if y in capex}

    ta = by_year(get_row(bal, ROWS["total_assets"]))
    tl = by_year(get_row(bal, ROWS["total_liab"]))
    eq = by_year(get_row(bal, ROWS["equity"]))
    ca = by_year(get_row(bal, ROWS["curr_assets"]))
    cl = by_year(get_row(bal, ROWS["curr_liab"]))
    debt = by_year(get_row(bal, ROWS["total_debt"]))
    ltd = by_year(get_row(bal, ROWS["lt_debt"]))
    cash = by_year(get_row(bal, ROWS["cash"]))
    re_ = by_year(get_row(bal, ROWS["retained"]))
    sh_b = by_year(get_row(bal, ROWS["shares"]))
    recv = by_year(get_row(bal, ROWS["receivables"]))
    icap = by_year(get_row(bal, ROWS["invested_cap"]))
    wc = by_year(get_row(bal, ROWS["working_cap"]))

    # ── SEC EDGAR BİRİNCİL (temel akış kalemleri) ─────────────────────
    # Rapor bulgusu: EDGAR canlı API'si <1 dk gecikmeli; Yahoo temel verisi
    # günler geride kalabilir ve yalnız ~4 yıl verir. Bu yüzden gelir /
    # net kâr / faaliyet kârı / CFO / CapEx / özsermaye YILLIK serilerinde
    # SEC değeri ESAS alınır: çakışan yılda SEC kazanır, SEC'in ekstra
    # yılları seriye eklenir (10 yıla kadar derinlik). Bilanço detayı
    # (borç, dönen varlık vb.) Yahoo'da kalır ve SEC çaprazıyla denetlenir.
    _ey = ((b.get("edgar") or {}).get("yillik") or {})
    m["temel_kaynak"] = "Yahoo"
    if _ey:
        def _sec_esas(yahoo_d, ad):
            e = _ey.get(ad) or {}
            if not e:
                return yahoo_d
            out_ = dict(yahoo_d or {})
            out_.update({int(y): float(v) for y, v in e.items()
                         if v is not None})
            return out_
        rev = _sec_esas(rev, "gelir")
        ni = _sec_esas(ni, "net_kar")
        oi = _sec_esas(oi, "faaliyet_kari")
        cfo = _sec_esas(cfo, "cfo")
        capex = _sec_esas(capex, "capex")
        eq = _sec_esas(eq, "ozsermaye")
        fcf = dict(fcf or {})
        for _y in cfo:                        # SEC derinliğiyle FCF uzat
            if _y in capex and _y not in fcf:
                fcf[_y] = cfo[_y] - abs(capex[_y])
        m["temel_kaynak"] = ("SEC EDGAR birincil (gelir/kâr/CFO/CapEx/"
                             "özsermaye) + Yahoo tamamlayıcı")

    for k, v in (("rev", rev), ("gp", gp), ("oi", oi), ("ni", ni),
                 ("ebitda_s", ebitda_s), ("cfo", cfo), ("capex", capex),
                 ("sbc", sbc), ("fcf", fcf)):
        m[k] = v
    m["years"] = sorted(rev.keys()) if rev else sorted(fcf.keys())

    # --- hisse, piyasa değeri, bilanço anları -----------------------------
    m["shares"] = (_num(info.get("sharesOutstanding"))
                   or b["fast"]["shares"] or latest(sh_b))
    m["mcap"] = _num(info.get("marketCap")) or b["fast"]["mcap"]
    if m["mcap"] is None and m["price"] and m["shares"]:
        m["mcap"] = m["price"] * m["shares"]

    m["total_debt"] = latest(debt) if debt else _num(info.get("totalDebt"))
    m["cash"] = latest(cash) if cash else _num(info.get("totalCash"))
    # Net borç: borç verisi YOKKEN "borç = 0" varsaymak iyimser bir uydurma
    # olur — yalnız nakit biliniyorsa None bırakılır (sanity notu düşer).
    m["net_debt"] = ((m["total_debt"] - (m["cash"] or 0))
                     if m["total_debt"] is not None else None)
    m["equity"] = latest(eq)
    m["eq_seri"] = eq          # FIX: SEC mutabakatı için yıllık özsermaye serisi
    # ── EV KÖPRÜSÜ (araştırma B2): standart köprü
    # EV = PD + Net Borç + İmtiyazlı + Azınlık Payı (+ emeklilik açığı)
    _nci = latest(by_year(get_row(bal, ["Minority Interest",
                                        "Minority Interests"])))
    if _nci is None:                       # yedek: SEC yıllık serisi
        _e_nci = ((b.get("edgar") or {}).get("yillik") or {}
                  ).get("azinlik_payi")
        if isinstance(_e_nci, dict) and _e_nci:
            _nci = latest(_e_nci)
    # MAKULLÜK BEKÇİSİ (TUR3): azınlık payı piyasa değerinin YARISINI
    # aşıyorsa kalem şüphelidir (yanlış etiket) — EV'ye eklenmez.
    if _nci is not None and m.get("mcap") and _nci > 0.5 * m["mcap"]:
        _nci = None
    _pref = latest(by_year(get_row(bal, ["Preferred Stock",
                                         "Preferred Securities Outside "
                                         "Stock Equity"])))
    # FAALİYET KİRASI (ASC 842) — denetim eksiği. Kiralar 2019'dan beri
    # bilançoda ve borç-benzeridir; EV köprüsüne katılmazsa kira-yoğun
    # şirketlerde (perakende, havayolu, restoran) EV düşük görünür.
    # Rapor G7: UV + KV ayrı satırlar TOPLANIR (get_row'un 'içerir'
    # eşleşmesi tek satırda kalabiliyordu); ikisi de yoksa toplam denenir.
    _k_uv = latest(by_year(get_row(bal,
                   ["Long Term Operating Lease Liability"])))
    _k_kv = latest(by_year(get_row(bal,
                   ["Current Operating Lease Liability"])))
    if _k_uv is not None or _k_kv is not None:
        _kira = (_k_uv or 0.0) + (_k_kv or 0.0)
    else:
        _kira = latest(by_year(get_row(bal,
                       ["Operating Lease Liability"])))
    # EMEKLİLİK AÇIĞI (rapor G8): ASC 715 sonrası fonlanma durumu (PBO −
    # plan varlıkları) NET YÜKÜMLÜLÜK olarak bilançodadır; borç-benzeridir.
    # Ağır DB-planlı sanayi/havayolu şirketlerinde EV'yi düzeltir.
    _emekli = latest(by_year(get_row(bal, [
        "Pension And Other Post Retirement Benefit Plans Total",
        "Non Current Pension And Other Postretirement Benefit Plans",
        "Pensionand Other Postretirement Benefit Plans Current",
        "Pension And Other Postretirement Benefit Plans"])))
    if _emekli is not None and _emekli <= 0:
        _emekli = None                 # fazla fonlanmışsa EV'ye ekleme
    m["ev_kira"] = _kira
    m["ev_emekli"] = _emekli
    m["ev_nci"] = _nci
    m["ev_imtiyazli"] = _pref
    m["ev"] = _num(info.get("enterpriseValue"))
    _ev_kalem = []
    if m["ev"] is None and m["mcap"] is not None:
        m["ev"] = m["mcap"] + (m["net_debt"] or 0)
        for _ad, _v in (("azınlık payı", _nci), ("imtiyazlı hisse", _pref),
                        ("faaliyet kirası (ASC 842)", _kira),
                        ("emeklilik açığı (net yükümlülük)", _emekli)):
            if _v and _v > 0:
                m["ev"] += _v
                _ev_kalem.append(_ad)
    m["ev_kopru_notu"] = (
        ("EV köprüsüne eklendi: " + ", ".join(_ev_kalem) + "."
         if _ev_kalem else None)
        if m.get("mcap") else None)

    ca0, cl0 = latest(ca), latest(cl)
    m["current_ratio"] = (ca0 / cl0) if (ca0 is not None and cl0) else None

    m["ebitda_ttm"] = _num(info.get("ebitda")) or latest(ebitda_s)
    m["nd_ebitda"] = ((m["net_debt"] / m["ebitda_ttm"])
                      if (m["net_debt"] is not None and m["ebitda_ttm"]
                          and m["ebitda_ttm"] > 0) else None)

    ebit0 = latest(by_year(get_row(inc, ROWS["ebit"]))) or latest(oi)
    m["ebit_latest"] = ebit0
    ie = latest(by_year(get_row(inc, ROWS["interest_exp"])))
    m["int_exp"] = abs(ie) if ie is not None else None
    m["int_cov"] = ((ebit0 / m["int_exp"])
                    if (ebit0 is not None and m["int_exp"]) else None)

    # --- TTM FCF & SBC ----------------------------------------------------
    # SEC ÇEYREKLİK BİRİNCİL: EDGAR'dan kurulan TTM, Yahoo çeyrekliklerine
    # tercih edilir (Yahoo yalnız ~4-5 çeyrek verir ve gecikebilir).
    _ec = ((b.get("edgar") or {}).get("ceyreklik") or {})
    _sec_cfo = ttm_sec(_ec, "cfo")
    _sec_capex = ttm_sec(_ec, "capex")
    ttm_fcf, _ttm_kaynak = None, None
    if _sec_cfo is not None and _sec_capex is not None:
        ttm_fcf = _sec_cfo - abs(_sec_capex)
        _ttm_kaynak = "SEC EDGAR çeyreklikleri (birincil)"
    if ttm_fcf is None:
        ttm_fcf = ttm_sum(qcf, ROWS["fcf"])
        if ttm_fcf is None:
            t_cfo, t_cap = (ttm_sum(qcf, ROWS["cfo"]),
                            ttm_sum(qcf, ROWS["capex"]))
            if t_cfo is not None and t_cap is not None:
                ttm_fcf = t_cfo - abs(t_cap)
        if ttm_fcf is not None:
            _ttm_kaynak = "Yahoo çeyreklikleri (yedek)"
    m["ttm_fcf_raw"] = ttm_fcf if ttm_fcf is not None else latest(fcf)
    m["ttm_kaynak"] = _ttm_kaynak
    m["fcf_source"] = ((f"TTM — {_ttm_kaynak}") if ttm_fcf is not None
                       else "Son yıllık" if latest(fcf) is not None else "YOK")
    # Rapor B4 etiketi: bu akış FCFE DEĞİLDİR — dürüst kimlik.
    m["dcf_akis_kimligi"] = (
        "CFO − CapEx: KALDIRAÇLI, net-borçlanma-HARİÇ özsermaye akışı "
        "(FCFE-benzeri ama FCFE DEĞİL — gerçek FCFE net borçlanmayı da "
        "içerir). CAPM ile iskonto edildiği için sonuç özsermaye "
        "değeridir; borç seviyesi hızla değişen şirkette bu akış "
        "kaldıraç değişimini YAKALAMAZ, temkinli oku.")
    m["ttm_ni"] = ttm_sec(_ec, "net_kar")
    m["ttm_rev"] = ttm_sec(_ec, "gelir")
    m["ttm_sbc"] = ttm_sum(qcf, ROWS["sbc"]) or latest(sbc)
    m["sbc_yontem"] = (
        "SBC nakit akışından DÜŞÜLDÜ (gelecek seyreltme) + oranlarda "
        "SEYRELTİLMİŞ hisse (geçmiş seyreltme) — Damodaran/Nixon: bu ikisi "
        "farklı dönemleri yakalar, çifte sayım DEĞİLDİR."
        if a.get("sbc_adjust") else
        "SBC nakit akışından düşülmedi — bu durumda seyreltilmiş hisse "
        "kullanmak ŞART (aksi hâlde seyreltme hiç yakalanmaz).")

    m["fcf_base"] = m["ttm_fcf_raw"]
    if a["sbc_adjust"] and m["fcf_base"] is not None and m["ttm_sbc"]:
        m["fcf_base"] = m["fcf_base"] - m["ttm_sbc"]

    # --- büyüme & marjlar -------------------------------------------------
    m["rev_cagr3"] = cagr(rev, 3)
    m["fcf_cagr3"] = cagr(fcf, 3)
    m["earnings_growth_info"] = _num(info.get("earningsGrowth"))
    m["gm"] = {y: gp[y] / rev[y] for y in gp if y in rev and rev[y]}
    # NOVY-MARX BRÜT KÂRLILIK (denetim eksiği): brüt kâr / toplam varlık.
    # Literatürde tek başına en güçlü kalite metriği; net kârdan daha az
    # muhasebe müdahalesine açıktır (Novy-Marx 2013, "The Other Side of
    # Value"). Yüksek = üretken varlık tabanı.
    _gp0, _ta0 = latest(gp), latest(ta)
    m["gpoa"] = (_gp0 / _ta0) if (_gp0 is not None and _ta0) else None
    m["gpoa_yorum"] = (
        None if m["gpoa"] is None else
        ("güçlü (>0.33)" if m["gpoa"] > 0.33 else
         "ortalama (0.15-0.33)" if m["gpoa"] >= 0.15 else "zayıf (<0.15)"))
    m["om"] = {y: oi[y] / rev[y] for y in oi if y in rev and rev[y]}
    m["nm"] = {y: ni[y] / rev[y] for y in ni if y in rev and rev[y]}
    rnd = by_year(get_row(inc, ROWS["rnd"]))
    m["rnd_rev"] = {y: rnd[y] / rev[y] for y in rnd if y in rev and rev[y]}
    m["sbc_rev_seri"] = {y: sbc[y] / rev[y] for y in sbc if y in rev and rev[y]}

    ni0, eq0, eq1 = latest(ni), latest(eq), latest(eq, 1)
    # ROE = net kâr / ORTALAMA özsermaye (CFA standardı). Negatif özsermaye
    # oranı anlamsız kılar → None (saçma sayı ASLA gösterilmez; sanity notu
    # ayrıca düşer).
    if eq0 is not None and eq0 > 0:
        eq_avg = ((eq0 + eq1) / 2 if (eq1 is not None and eq1 > 0) else eq0)
        m["roe"] = (ni0 / eq_avg) if ni0 is not None else None
        m["roe_basis"] = ("2Y ortalama özsermaye"
                          if (eq1 is not None and eq1 > 0)
                          else "dönem sonu özsermaye (tek yıl var)")
    else:
        m["roe"] = None
        m["roe_basis"] = ("özsermaye negatif — ROE anlamsız"
                          if eq0 is not None else None)

    # ROIC — DOĞRU formül: NOPAT / Yatırılan Sermaye. ROE ile ASLA karışmaz.
    # NOPAT = EBIT × (1 − EFEKTİF vergi); vergi tabloda ne yazıyorsa o,
    # sabit %21 varsayımı değil.
    tax0 = latest(by_year(get_row(inc, ROWS["tax"])))
    ptx0 = latest(by_year(get_row(inc, ROWS["pretax"])))
    eff_tax = (tax0 / ptx0) if (tax0 is not None and ptx0 and ptx0 > 0) else None
    if eff_tax is None or not (0.0 <= eff_tax <= 0.45):
        eff_tax = 0.21                       # makul aralık dışıysa yasal oran
    # ── NAKİT VERGİ DÜZELTMESİ (Damodaran) ────────────────────────────
    # NOPAT/ROIC'te doğru gider DEFTER vergisi değil ÖDENEN vergidir;
    # ertelenmiş vergi bugün nakit çıkışı yaratmaz. Fark büyük şirketlerde
    # tipik olarak küçüktür (~%5) ama hızlandırılmış amortisman veya yüksek
    # SBC kullanan şirketlerde açılır. Sıra: (1) nakit akışındaki ödenen
    # vergi satırı, (2) defter vergisi − ertelenmiş vergi, (3) defter.
    _nakit_vergi, _vergi_kaynak = None, "defter vergisi (tablo)"
    try:
        _tp = latest(by_year(get_row(cf, ROWS["tax_paid"])))
        if _tp is not None and abs(_tp) > 0:
            _nakit_vergi, _vergi_kaynak = abs(_tp), "ödenen vergi (nakit akışı)"
        else:
            _td = latest(by_year(get_row(cf, ROWS["tax_deferred"])))
            if _td is not None and tax0 is not None:
                _nakit_vergi = abs(tax0) - _td
                _vergi_kaynak = "defter vergisi − ertelenmiş vergi"
    except Exception:
        _nakit_vergi = None
    m["eff_tax_defter"] = round(eff_tax, 4)
    if _nakit_vergi is not None and ptx0 and ptx0 > 0:
        _et = _nakit_vergi / ptx0
        if 0.0 <= _et <= 0.45:               # makul bant dışı → defterde kal
            eff_tax = _et
        else:
            _vergi_kaynak = (f"nakit vergi oranı %{_et*100:.0f} makul bant "
                             f"dışı → defter vergisi kullanıldı")
    m["eff_tax"] = eff_tax
    m["vergi_kaynak"] = _vergi_kaynak
    m["nakit_vergi_musd"] = (round(_nakit_vergi / 1e6, 1)
                             if _nakit_vergi is not None else None)
    ic_ys = sorted(icap)
    if len(ic_ys) >= 2:                      # dönem ortalaması (başı+sonu)/2
        ic0 = (icap[ic_ys[-1]] + icap[ic_ys[-2]]) / 2
    elif ic_ys:
        ic0 = icap[ic_ys[-1]]
    else:                                    # bilanço yolu: özkaynak+borç−nakit
        ic0 = ((eq0 or 0) + (m["total_debt"] or 0) - (m["cash"] or 0)
               if eq0 is not None else None)
    # ── FAALİYET KİRASI KAPİTALİZASYONU — ROIC (rapor G7, Damodaran):
    # kira yükümlülüğü YATIRILAN SERMAYEYE, kiranın gömülü FAİZ bileşeni
    # (≈ kira borcu × borç maliyeti) EBIT'e eklenir. Kira gideri EBIT'te
    # TAM gider yazıldığı için finansman payını geri eklemek çifte sayım
    # DEĞİLDİR. Borç maliyeti: faiz gideri / toplam borç, [%2, %12]
    # bandına kırpılır; veri yoksa Rf + 150bp. [TÜRETİLMİŞ ~]
    ebit_roic = ebit0
    m["roic_kira_detay"] = None
    if _kira and _kira > 0 and ic0 and ic0 > 0 and ebit0 is not None:
        _kd_ = ((m["int_exp"] / m["total_debt"])
                if (m.get("int_exp") and m.get("total_debt")) else None)
        if _kd_ is None or not (0.02 <= _kd_ <= 0.12):
            # Bu satır HAM SABİTİ okuyordu → kenar çubuğundan Rf
            # değiştirilse bile burası ESKİ değerde kalıyordu.
            _kd_ = rf_erp_guncel()[0] + 0.015
        _kfaiz = _kira * _kd_
        ic0 = ic0 + _kira
        ebit_roic = ebit0 + _kfaiz
        m["roic_kira_detay"] = {
            "kira_yukumlulugu_musd": round(_kira / 1e6, 1),
            "gomulu_faiz_musd": round(_kfaiz / 1e6, 1),
            "borc_maliyeti": round(_kd_, 4),
            "not": ("Kira yükümlülüğü sermayeye, gömülü faiz EBIT'e "
                    "eklendi (Damodaran, ASC 842) [TÜRETİLMİŞ ~].")}
    nopat = ebit_roic * (1 - eff_tax) if ebit_roic is not None else None
    m["roic"] = (nopat / ic0) if (nopat is not None and ic0 and ic0 > 0) else None
    m["roic_detay"] = {
        "formul": "EBIT(+kira faizi)×(1−efektif vergi) / "
                  "Yat. Sermaye (2Y ort. + kira yükümlülüğü)",
        "efektif_vergi": round(eff_tax, 4),
        "nopat_musd": round(nopat / 1e6, 1) if nopat is not None else None,
        "yatirilan_sermaye_musd": round(ic0 / 1e6, 1) if ic0 else None,
    }
    m["roic_ham"] = m["roic"]          # düzeltme öncesi ham değer (şeffaflık)

    # ── AR-GE KAPİTALİZASYONU: ROIC'i düzelt (Damodaran) ──────────────
    # Ar-Ge/Gelir eşiği %4: altındaysa şirket Ar-Ge-yoğun değildir, düzeltme
    # gürültü katar. Üstündeyse DÜZELTİLMİŞ ROIC KARAR DEĞERİ olur — çünkü
    # m["roic"] hem kalite skorunu (ağırlık 2.5) hem oto_buyume'deki DCF
    # büyüme TAVANINI besliyor; şişmiş ROIC ikisini birden iyimser yapıyordu.
    m["roic_arge"] = None
    m["roic_kaynak"] = "ham (Ar-Ge düzeltmesi uygulanmadı)"
    try:
        _rr = latest(m.get("rnd_rev") or {})
        if _rr is not None and _rr >= 0.04 and nopat is not None:
            _ra = roic_arge_kapitalize(
                rnd, ebit_roic, ic0, eff_tax,
                omur=arge_omru(m.get("sector"), m.get("industry")))
            if _ra:
                m["roic_arge"] = _ra
                m["roic"] = _ra["roic_duzeltmeli"]      # KARAR DEĞERİ
                m["roic_kaynak"] = (
                    f"Ar-Ge kapitalize (ömür {_ra['etkin_omur_yil']}y): "
                    f"ham {_ra['roic_ham']*100:.1f}% → "
                    f"{_ra['roic_duzeltmeli']*100:.1f}% "
                    f"({_ra['fark_puan']:+.1f} puan)")
                m["roic_detay"]["arge_duzeltmesi"] = m["roic_kaynak"]
    except Exception:
        pass

    # ── BAKIM CAPEX AYRIMI (Greenwald) — EPV bunu kullanır ────────────
    try:
        # B5-1: brüt PP&E varsa o kullanılır; yoksa net'e düşer ama bu
        # durum bakim_capex içinde AÇIKÇA etiketlenir.
        _ppe_b = (by_year(get_row(bal, ROWS["ppe_brut"]))
                  if bal is not None else {})
        _ppe_n = (by_year(get_row(bal, ROWS["ppe"]))
                  if bal is not None else {})
        m["capex_ayrim"] = bakim_capex(
            rev, capex, (_ppe_b if len(_ppe_b) >= 2 else _ppe_n),
            by_year(get_row(cf, ROWS["dep"])) if cf is not None else {})
        if m["capex_ayrim"] is not None:
            m["capex_ayrim"]["ppe_bazi"] = (
                "brüt PP&E (Greenwald standardı)" if len(_ppe_b) >= 2 else
                "NET PP&E — brüt seri yok; bakım capex OLDUĞUNDAN BÜYÜK, "
                "dolayısıyla EPV OLDUĞUNDAN DÜŞÜK olabilir")
    except Exception:
        m["capex_ayrim"] = None
    m["mcap_info"] = _num(info.get("marketCap"))
    m["eps_info"] = _num(info.get("trailingEps"))

    # --- çarpanlar --------------------------------------------------------
    m["pe_t"] = _num(info.get("trailingPE"))
    m["pe_f"] = _num(info.get("forwardPE"))
    m["ps"] = _num(info.get("priceToSalesTrailing12Months"))
    m["pb"] = _num(info.get("priceToBook"))
    m["ev_ebitda"] = _num(info.get("enterpriseToEbitda"))
    if m["ev_ebitda"] is None and m["ev"] and m["ebitda_ttm"] and m["ebitda_ttm"] > 0:
        m["ev_ebitda"] = m["ev"] / m["ebitda_ttm"]
    m["p_fcf"] = ((m["mcap"] / m["ttm_fcf_raw"])
                  if (m["mcap"] and m["ttm_fcf_raw"] and m["ttm_fcf_raw"] > 0)
                  else None)
    m["p_fcf_adj"] = ((m["mcap"] / m["fcf_base"])
                      if (m["mcap"] and m["fcf_base"] and m["fcf_base"] > 0)
                      else None)
    m["peg"], m["peg_src"] = resolve_peg(info, m["pe_t"])

    # --- kalite & adli göstergeler ----------------------------------------
    m["piotroski"] = piotroski_f(ni, cfo, ta, ltd, ca, cl, sh_b, gp, rev)
    # KANIT TURU (araştırma C2): Piotroski (2000) KÜÇÜK-CAP + YÜKSEK
    # book-to-market (değer) evreninde çalışıldı; büyük-cap ve düşük B/M'de
    # geçerliliği zayıf, 2006 sonrası bazı çalışmalarda NEGATİF. Skoru
    # değiştirmiyoruz; evren dışıysa etiketliyoruz.
    _bm = (1.0 / m["pb"]) if m.get("pb") else None
    if (m.get("mcap") or 0) > 1e10 or (_bm is not None and _bm < 0.5):
        m["piotroski_evren_notu"] = (
            "EVREN DIŞI: F-skoru küçük-cap + yüksek book-to-market (değer) "
            "evreninde doğrulandı. Bu şirket büyük-cap ve/veya düşük B/M — "
            "burada F-skoru DESTEKLEYİCİ bir kalite göstergesidir, alım "
            "sinyali değildir. [Piotroski 2000]")
    else:
        m["piotroski_evren_notu"] = None
    _indl = (m["industry"] or "").lower()
    m["is_financial"] = (m["sector"] == "Financial Services"
                         or any(k in _indl for k in
                                ("bank", "insurance", "capital market",
                                 "credit services", "asset management")))
    if m["is_financial"] and m.get("ev_ebitda") is not None:
        # Bankada faiz OPERASYONEL kalemdir → EBITDA/EV anlamsızdır; yanlış
        # sinyal vermemek için çarpan gizlenir (P/E ve P/B esastır).
        m["ev_ebitda"] = None
        m["ev_ebitda_note"] = ("finansal sektörde anlamsız olduğu için "
                               "gizlendi — bankada faiz faaliyet kalemidir; "
                               "P/E ve P/B'ye bak")
        m["banka_dcf_uyarisi"] = (
            "FİNANSAL ŞİRKET: CFO−CapEx tabanlı FCF ve DCF bankada kavramsal "
            "olarak ZAYIFTIR (borç hammaddedir). Değerlemeyi P/TBV–ROTCE "
            "merceğiyle oku; buradaki DCF yalnız kaba bağlamdır.")
    wc0 = latest(wc)
    if wc0 is None and ca0 is not None and cl0 is not None:
        wc0 = ca0 - cl0
    m["altman"] = None
    if not m["is_financial"]:
        _sekt = (m.get("sector") or "").lower()
        _uretim_mi = any(k in _sekt for k in
                         # O5: perakende/GYO/kamu hizmeti üretici DEĞİL — imalat-dışına
                         # Altman Z'' önerir; eski liste onları X5'li orijinal Z'ye sokuyordu.
                         ("industrial", "material", "energy"))
        _mdl = "uretim" if _uretim_mi else "hizmet"
        m["altman"], m["altman_model"] = altman_z(
            wc0, latest(re_), ebit0, m["mcap"], latest(tl),
            latest(rev), latest(ta), model=_mdl, bve=m.get("equity"))
    else:
        m["altman_model"] = None
    ta0, cfo0 = latest(ta), latest(cfo)
    m["accrual"] = ((ni0 - cfo0) / ta0
                    if None not in (ni0, cfo0) and ta0 else None)
    m["cfo_lt_ni_2y"] = None
    if all(latest(x, k) is not None for x in (cfo, ni) for k in (0, 1)):
        m["cfo_lt_ni_2y"] = (latest(cfo, 0) < latest(ni, 0)
                             and latest(cfo, 1) < latest(ni, 1))

    m["dso"] = {y: recv[y] / rev[y] * 365 for y in recv if y in rev and rev[y]}
    # DİLÜSYON — DOĞRU ölçüm: yıllık ORTALAMA seyreltilmiş hisse (gelir
    # tablosundan) değişimi. Dönem sonu hisseyle ölçmek, IPO/tercihli hisse
    # dönüşümü yılında tek seferlik sıçramayı 'sürekli dilüsyon' sanma
    # tuzağıdır — o tuzak burada kapatıldı.
    dsh = by_year(get_row(inc, ROWS["dil_shares"]))
    if not dsh:
        dsh = dict(sh_b)                     # yedek: dönem sonu serisi
    _e_dsh = ((b.get("edgar") or {}).get("yillik") or {}
              ).get("seyreltilmis_hisse") or {}
    if _e_dsh:                               # SEC birincil: hisse serisi
        dsh = dict(dsh or {})
        dsh.update({int(y): float(v) for y, v in _e_dsh.items() if v})
    m["dil_shares_seri"] = dsh
    m["dilution_yoy"] = None                 # temiz yıllık dilüsyon (medyan)
    m["dilution_seri"] = {}
    m["dilution_ipo_jump"] = None
    d_ys = sorted(dsh)
    if len(d_ys) >= 2:
        chg = {d_ys[i]: dsh[d_ys[i]] / dsh[d_ys[i - 1]] - 1
               for i in range(1, len(d_ys)) if dsh[d_ys[i - 1]]}
        m["dilution_seri"] = chg
        jump = {y: c for y, c in chg.items() if abs(c) > 0.60}
        clean = sorted(c for c in chg.values() if abs(c) <= 0.60)
        if jump:
            m["dilution_ipo_jump"] = {str(y): round(c, 4)
                                      for y, c in jump.items()}
        if clean:
            m["dilution_yoy"] = _medyan(clean)    # BULGU U
    rev0 = latest(rev)
    m["sbc_rev"] = ((m["ttm_sbc"] / rev0)
                    if (m["ttm_sbc"] is not None and rev0) else None)

    # ── GERİ ALIM KALİTESİ: iade mi, SBC telafisi mi? ─────────────────
    # SIRA NOTU: bu blok bilerek dilüsyon hesabından SONRA duruyor —
    # m["dilution_yoy"] girdisi yukarıda üretiliyor. Daha yukarıda
    # çağrılırsa gözlenen net seyreltme her zaman None geçer (sessiz
    # ölü değer). Okuyucuları (build_red_flags, capital_allocation)
    # çok daha aşağıda olduğu için burası güvenli.
    try:
        m["geri_alim"] = geri_alim_kalitesi(
            m.get("ttm_sbc") or latest(sbc),
            latest(by_year(get_row(cf, ROWS["buyback"]))),
            m.get("mcap"), m.get("dilution_yoy"))
    except Exception:
        m["geri_alim"] = None
    # Rapor bulgusu: forwardPE/pegRatio gibi .info türev alanları hatalı/
    # eksik olabiliyor (yfinance #1911/#2570). Trailing F/K boşsa kendi
    # hesabımız devreye girer: fiyat / (net kâr ÷ ort. seyreltilmiş hisse).
    if m["pe_t"] is None and m.get("price") and dsh:
        _d0 = dsh.get(max(dsh))
        # Denetim Bulgu 5: SEC TTM net kâr hesaplanıyordu ama burada
        # YILLIK NI kullanılıyordu — daha taze veri bitişikte boşta
        # duruyordu. Artık önce TTM, yoksa yıllık.
        _ni_pe = (m.get("ttm_ni") if (m.get("ttm_ni") or 0) > 0 else
                  (ni0 if (ni0 or 0) > 0 else None))
        if _d0 and _ni_pe:
            m["pe_t"] = m["price"] / (_ni_pe / _d0)
            m["pe_t_kaynak"] = (
                "kendi hesap (SEC TTM NI ÷ ort. seyreltilmiş hisse)"
                if _ni_pe == m.get("ttm_ni") else
                "kendi hesap (yıllık NI ÷ ort. seyreltilmiş hisse)")

    # --- DCF, ters-DCF, duyarlılık ----------------------------------------
    if a.get("oto_buyume", True):
        _g, _gnot = oto_buyume(m, a["growth"])
        a = dict(a, growth=_g)
        m["buyume_kaynak"] = _gnot
        m["buyume_kullanilan"] = _g
    else:
        m["buyume_kaynak"] = "elle ayarlandı"
        m["buyume_kullanilan"] = a["growth"]
    # ── ARAŞTIRMA TURU A2 / B4-2 BAĞLANDI (bu tarihe kadar ÖLÜ KODDU) ──
    # roic_terminal parametresi tanımlıydı ama HİÇBİR çağrı geçirmiyordu;
    # terminal reinvestman kısıtı fiilen kapalıydı (denetim Bulgu 3).
    # Artık TÜM motorlar (fair + bant + duyarlılık + ters-DCF + MC) aynı
    # kısıtla çalışır — biri açık biri kapalı olursa sayılar kıyaslanamaz.
    _rt = terminal_roic_sec(m.get("roic"), a["discount"])
    a = dict(a, roic_terminal=_rt)
    m["roic_terminal_kullanilan"] = _rt
    m["terminal_reinvest_notu"] = (
        None if _rt is None else
        ("Terminal yeniden-yatırım kısıtı AKTİF: RONIC "
         f"%{_rt*100:.1f} "
         + ("(= iskonto — rekabetçi denge; moat kanıtı yok/ROIC≤iskonto)"
            if abs(_rt - a["discount"]) < 1e-9 else
            "(ROIC−iskonto priminin yarısı, +5 puan tavan — moat vekili)")
         + ". O3: ×(1−g/RONIC) çarpanı FCF tabanında ÇİFTE SAYIM olduğu "
           "için kaldırıldı (FCF yeniden-yatırımı zaten düşer; McKinsey "
           "formülü NOPAT tabanına aittir). RONIC yalnız bilgi amaçlı."))
    m["fair"] = dcf_fair_value(m["fcf_base"], a["growth"], a["discount"],
                               a["terminal"], a["years"], m["net_debt"],
                               m["shares"], a["net_debt_adjust"],
                               roic_terminal=a.get("roic_terminal"))
    # ADİL DEĞER TEK SAYI DEĞİLDİR: kötümser/iyimser senaryo bandı da hesapla.
    g_p = max(a["growth"] - 0.04, -0.05)
    d_p = min(a["discount"] + 0.015, 0.18)
    g_o = min(a["growth"] + 0.03, 0.35)
    d_o = max(a["discount"] - 0.01, a["terminal"] + 0.015)
    m["fair_scn"] = {"kotumser": {"g": g_p, "d": d_p},
                     "iyimser": {"g": g_o, "d": d_o}}
    lo_f = dcf_fair_value(m["fcf_base"], g_p, d_p, a["terminal"], a["years"],
                          m["net_debt"], m["shares"], a["net_debt_adjust"],
                          roic_terminal=a.get("roic_terminal"))
    hi_f = dcf_fair_value(m["fcf_base"], g_o, d_o, a["terminal"], a["years"],
                          m["net_debt"], m["shares"], a["net_debt_adjust"],
                          roic_terminal=a.get("roic_terminal"))
    if lo_f is not None and hi_f is not None and lo_f > hi_f:
        lo_f, hi_f = hi_f, lo_f

    # ── B4-6 BAĞLANDI (bu tarihe kadar ÖLÜ KODDU) ───────────────────
    # İflas riski, iskonto oranına prim eklenerek DEĞİL, değere KESİNTİ
    # olarak yansıtılır (Damodaran): iflas ikili bir olaydır, nakit
    # akışını sürekli daha çok iskonto etmek onu doğru modellemez.
    # distres_kesintisi() yazılmış ama HİÇBİR YERDEN ÇAĞRILMAMIŞTI →
    # Altman Z'si düşük şirketler kesintisiz, TAM değerleniyordu.
    # Kesinti üç değere de uygulanır (nokta + bant), yoksa bant nokta
    # tahminle tutarsız kalır.
    m["distres"] = None
    try:
        # ARAŞTIRMA TURU A1: eşik artık MODELE göre — Z'' hizmet skoru
        # üretim eşikleriyle kesilmiyor (denetim Bulgu 1a).
        _kes = distres_kesintisi(m.get("altman"), m.get("is_financial"),
                                 model=m.get("altman_model"))
    except Exception:
        _kes = None
    if _kes and _kes[0]:
        _o = float(_kes[0])
        for _k in ("fair",):
            if m.get(_k):
                m[_k] = m[_k] * (1.0 - _o)
        if lo_f is not None:
            lo_f *= (1.0 - _o)
        if hi_f is not None:
            hi_f *= (1.0 - _o)
        m["distres"] = {"oran": _o, "gerekce": _kes[1]}
    m["fair_lo"], m["fair_hi"] = lo_f, hi_f
    m["mos"] = ((m["fair"] - m["price"]) / m["fair"]
                if (m["fair"] and m["price"]) else None)
    m["mc"] = monte_carlo_dcf(m["fcf_base"], a, m["net_debt"], m["shares"],
                              a["net_debt_adjust"], m["price"])
    # Aşama 1 — kalite sürücüleri + normalize değerleme + senaryo + tuzaklar
    m["_inc"], m["_bal"] = inc, bal
    m["dupont"] = dupont(m)
    m["epv"] = epv_value(m, a)
    m["ccc"] = cash_cycle(m)
    m["scenario"] = scenario_ev(m)      # mc'ye bağımlı — mc'den sonra
    m["implied_g"], m["implied_capped"] = implied_growth(
        m["price"], m["fcf_base"], a["discount"], a["terminal"], a["years"],
        m["net_debt"], m["shares"], a["net_debt_adjust"],
        roic_terminal=a.get("roic_terminal"))
    m["sens"] = sensitivity(m["fcf_base"], a["growth"], a["discount"],
                            a["terminal"], a["years"], m["net_debt"],
                            m["shares"], a["net_debt_adjust"],
                            roic_terminal=a.get("roic_terminal"))

    # ── ARAŞTIRMA TURU A1-b (denetim Bulgu 1b): kesinti TÜM motorlara ──
    # Eski hâlde kesinti yalnız fair/lo/hi'ye iniyordu; MC medyanı, EPV ve
    # duyarlılık matrisi KESİNTİSİZ kalıyor, "medyan nokta tahminin
    # etrafında oturur" notu distresli şirkette YANLIŞLAŞIYORDU. Artık
    # MC yüzdelikleri, dağılım, EPV ve duyarlılık matrisi aynı kesintiyi
    # taşır; senaryo-EV kesintili MC'den YENİDEN kurulur. Ters-DCF
    # bilerek KESİNTİSİZ kalır: o, fiyattan büyüme ÇIKARIMIDIR (going-
    # concern modelin tersi), değer tahmini değil.
    if (m.get("distres") or {}).get("oran"):
        _o2 = m["distres"]["oran"]
        _mc = m.get("mc")
        if _mc:
            for _k2 in ("p5", "p25", "p50", "p75", "p95"):
                if _mc.get(_k2) is not None:
                    _mc[_k2] = round(_mc[_k2] * (1 - _o2), 2)
            if _mc.get("dagilim"):
                _mc["dagilim"] = [round(x * (1 - _o2), 2)
                                  for x in _mc["dagilim"]]
                if m.get("price"):
                    _mc["yukari_potansiyel_olasiligi"] = round(
                        sum(1 for x in _mc["dagilim"]
                            if x > m["price"]) / len(_mc["dagilim"]), 3)
            _mc["not"] = ((_mc.get("not") or "")
                          + f" ⚠️ Tüm yüzdeliklere %{_o2*100:.1f} iflas "
                            f"kesintisi uygulandı — nokta tahminle aynı "
                            f"dünyada (bkz. iflas_distres_kesintisi).")
        _ep = m.get("epv")
        if _ep and _ep.get("epv_hisse_basi"):
            _ep["epv_hisse_basi"] = round(
                _ep["epv_hisse_basi"] * (1 - _o2), 2)
            if m.get("price") and _ep["epv_hisse_basi"] > 0:
                _ep["fiyata_gore"] = round(
                    m["price"] / _ep["epv_hisse_basi"] - 1, 3)
            _ep["not"] = ((_ep.get("not") or "")
                          + f" %{_o2*100:.1f} iflas kesintisi dahil.")
        if m.get("sens") and m["sens"].get("z"):
            m["sens"]["z"] = [[(None if v is None
                                else round(v * (1 - _o2), 2))
                               for v in row] for row in m["sens"]["z"]]
            m["sens"]["distres_notu"] = (f"matris %{_o2*100:.1f} iflas "
                                         f"kesintisi dahil")
        m["scenario"] = scenario_ev(m)   # kesintili MC'den yeniden kur
    # --- ÇELİŞKİ DETEKTÖRÜ girdileri: fiyat konumu & fiyat-değer makası ---
    hist = b.get("hist")
    close = (hist["Close"].dropna()
             if (hist is not None and "Close" in hist) else None)
    # 52 HAFTA — Yahoo'nun hazır değeri GÜVENİLMEZ olabiliyor (bilinen bug);
    # fiyat geçmişinden KENDİMİZ hesaplarız, info değeri sadece yedek.
    lo52_i = _num(info.get("fiftyTwoWeekLow"))
    hi52_i = _num(info.get("fiftyTwoWeekHigh"))
    lo52 = hi52 = None
    if close is not None and len(close) > 30:
        tail = close.iloc[-252:] if len(close) >= 252 else close
        lo52, hi52 = float(tail.min()), float(tail.max())
        if hist is not None and "High" in hist and "Low" in hist:
            th = hist["High"].dropna().iloc[-252:]
            tl = hist["Low"].dropna().iloc[-252:]
            if len(th) > 30:
                hi52 = float(th.max())
            if len(tl) > 30:
                lo52 = float(tl.min())
    if lo52 is None:
        lo52 = lo52_i
    if hi52 is None:
        hi52 = hi52_i
    m["lo52_info"], m["hi52_info"] = lo52_i, hi52_i
    m["lo52"], m["hi52"] = lo52, hi52
    m["pos_52w"] = ((m["price"] - lo52) / (hi52 - lo52)
                    if None not in (lo52, hi52, m["price"]) and hi52 > lo52
                    else None)

    m["price_chg_2y"] = None
    if close is not None and len(close) > 260 and m["price"]:
        span = min(len(close) - 1, 504)          # ~2 yıl işlem günü
        p_old = _num(close.iloc[-span - 1])
        if p_old and p_old > 0:
            m["price_chg_2y"] = m["price"] / p_old - 1
    ys_r = sorted(rev.keys())
    m["rev_chg_2y"] = (rev[ys_r[-1]] / rev[ys_r[-3]] - 1
                       if len(ys_r) >= 3 and rev[ys_r[-3]] > 0 else None)
    m["pv_gap"] = None                            # fiyat/temel makası (kat)
    if (m["price_chg_2y"] is not None and m["rev_chg_2y"] is not None
            and m["rev_chg_2y"] > 0.01 and m["price_chg_2y"] > 0):
        m["pv_gap"] = m["price_chg_2y"] / m["rev_chg_2y"]

    # Yaklaşık tarihsel F/S bandı (hisse sayısı bugünkü varsayılır →
    # geri alım/dilüsyon nedeniyle YAKLAŞIKTIR; etiketiyle sunulur)
    m["ps_band"] = None
    if close is not None and m.get("shares") and rev and m.get("ps"):
        band = {}
        for y, r in rev.items():
            yr = close[close.index.year == y]
            if len(yr) > 20 and r:
                band[y] = float(yr.mean()) * m["shares"] / r
        if len(band) >= 2:
            bmin, bmax = min(band.values()), max(band.values())
            pos = ((m["ps"] - bmin) / (bmax - bmin)) if bmax > bmin else None
            m["ps_band"] = {
                "yillik_ortalama": {str(y): round(v, 2)
                                    for y, v in sorted(band.items())},
                "guncel": round(m["ps"], 2),
                "konum_0dip_1zirve": (round(pos, 2)
                                      if pos is not None else None),
            }

    # --- Grup 2 katmanları: Beneish, katalizör, içeriden akış, revizyon ---
    m["beneish"] = None if m["is_financial"] else beneish_m(inc, bal, cf)
    m["olaylar_8k"] = b.get("olaylar_8k")
    m["haberler"] = b.get("haberler")
    m["catalysts"] = parse_catalysts(b)
    # Opsiyondan ima edilen hareket — varsa sıradaki kazanç tarihine bağla
    _ekz = None
    for _e in (m["catalysts"].get("gelecek") or []):
        if "Kazanç" in _e.get("olay", ""):
            _ekz = _e.get("tarih"); break
    try:
        m["implied_move"] = implied_move(b.get("ticker"), m.get("price"),
                                         _ekz)
    except Exception:
        m["implied_move"] = None
    _sec_l = None
    try:
        _cik = edgar_cik_map().get(str(b.get("ticker", "")).upper())
        if _cik:
            _sec_l = ("https://www.sec.gov/cgi-bin/browse-edgar?action="
                      f"getcompany&CIK={_cik:010d}&type=4&dateb=&owner="
                      "include&count=40")
    except Exception:
        _sec_l = None
    m["insider"] = parse_insider(b.get("insider_tx"), b.get("form4"),
                                 _sec_l)
    m["insider_link"] = _sec_l
    m["revizyon"] = parse_revisions(b.get("eps_trend"), b.get("eps_rev"),
                                    b.get("price_targets"), b.get("recs"),
                                    m["price"])
    m["short"] = parse_short(info)
    m["kurumsal"] = parse_holders(b.get("major_holders"),
                                  b.get("inst_holders"))
    m["ipo"] = parse_ipo(info, b.get("hist"))
    if (m["ipo"] or {}).get("durum") == "LOCKUP_GELECEK":
        gel = m["catalysts"]["gelecek"]
        gel.append({"olay": "IPO lock-up bitişi (tahminî, ~180g kuralı)",
                    "tarih": m["ipo"]["lockup_tahmini"],
                    "gun_kaldi": m["ipo"]["kalan_gun"]})
        m["catalysts"]["gelecek"] = sorted(gel,
                                           key=lambda e: e["gun_kaldi"])[:6]
    m["edgar"] = edgar_crosscheck(m, b.get("edgar"))

    # --- bayraklar, kalite, skor, hüküm -----------------------------------
    m["_cf"] = cf
    # FIX (sıra): F/K tarihsel bandı SKORA girdiği için forward_bands artık
    # skordan ÖNCE kurulur. Eskiden Aşama 3'te (skordan sonra) çağrılıyordu;
    # bu yüzden bant bileşeni her zaman boş kalıyordu.
    m["_edgar_yillik"] = (b.get("edgar") or {}).get("yillik")
    # B3-7: forward_bands bölünme çarpanını kullanır;
    # splits ONDAN ÖNCE dolmalı, yoksa çarpan hep 1
    # döner ve düzeltme SESSİZCE hiçbir şey yapmaz.
    m["splits"] = b.get("splits") or {}
    try:
        m["fwd"] = forward_bands(m, b)
    except Exception:
        m["fwd"] = None
    m["flags"] = build_red_flags(m) + forensic_flags(m)
    # ── SIRA DÜZELTMESİ (denetim Bulgu 4): bu blok eskiden Aşama 2'de,
    # kalite/skor/hüküm HESAPLANDIKTAN SONRA duruyordu. Sonuç: büyük-cap
    # F≤3 bayrağı ekranda/payload'da görünüyor ama kalite cezasına (−10),
    # skor primine (+4), skor kararlılığına ve hükme HİÇ işlemiyordu —
    # tam da en riskli kombinasyonda iki eksen iyimser kalıyordu.
    # PIOTROSKI KULLANIM SINIRI: F-Score, Piotroski (2000) tarafından
    # YÜKSEK book-to-market + küçük/orta-cap evreninde tanımlandı ve etki
    # orada güçlü. Büyük-cap'te skor bir "seçim kriteri" değil yalnız
    # KIRMIZI BAYRAK (F≤3) olarak yorumlanır.
    _pf = m.get("piotroski") or {}
    _buyuk = bool(m.get("mcap") and m["mcap"] > 1e10)
    m["piotroski_mod"] = "kirmizi_bayrak" if _buyuk else "secim"
    m["piotroski_notu"] = (
        ("Bu büyük ölçekli bir şirket. F-Score küçük/orta-cap DEĞER "
         "evreninde tanımlandı; burada seçim kriteri değil yalnız KIRMIZI "
         "BAYRAK olarak okunur — F≤3 ise dikkat, yüksek F tek başına "
         "'al' sinyali DEĞİLDİR.") if _buyuk else
        ("F-Score bu ölçekte tanımlandığı evrene yakın; seçim kriteri "
         "olarak kullanılabilir."))
    if _buyuk and (_pf.get("score") is not None) and _pf["score"] <= 3:
        m["flags"] = m["flags"] + [
            ("yüksek", f"Piotroski F={_pf['score']}/9 — finansal sağlıkta "
                       f"bozulma işareti (kırmızı bayrak eşiği ≤3).")]
    m["sanity"] = sanity_checks(m)
    m["quality"] = quality_score(m)
    m["score"] = expensiveness_score(m)
    # ── B9-5/6 BAĞLANDI (bu tarihe kadar ÖLÜ KODDU) ─────────────────
    # Skorun KENDİ belirsizlik bandı. fig_gauge bu alanı OKUYORDU ama
    # hiç kimse YAZMIYORDU → band hiç görünmedi.
    try:
        m["skor_kararlilik"] = skor_kararliligi(m)
    except Exception:
        m["skor_kararlilik"] = None
    m["verdict"] = verdict_info(m["score"], m["quality"])
    # SR/ATR, kademe'den ÖNCE hesaplanır ki giriş dilimleri destek
    # bölgeleriyle çapraz-teyit edilebilsin (rapor entegrasyon kararı).
    m["_hist"] = b.get("hist")
    # ── B3-8 BAĞLANDI (ÖLÜ KODDU) ───────────────────────────────────
    # Son bar kısmi (seans sürüyor) ya da gecikmeli olabilir; seviyeler
    # ve ATR ondan hesaplanıyor. Uyarı üretiliyordu ama gösterilmiyordu.
    try:
        m["son_bar"] = son_bar_durumu(m["_hist"])
    except Exception:
        m["son_bar"] = None
    _atr_ser = atr_serisi(m["_hist"], 22)
    m["atr22"] = (float(_atr_ser.iloc[-1])
                  if _atr_ser is not None and len(_atr_ser) else None)
    m["sr"] = destek_direnc(m["_hist"], m["price"], m["atr22"])
    # ── REJİM SİNYALLERİ (bütünsel rapor G4/G5): 12-1 momentum + PEAD.
    # kademeli_plan'dan ÖNCE hesaplanır ki plan TEMPO notlarını görsün.
    # İkisi de SKORA GİRMEZ — skor fiyat-değer sorusudur, bunlar zamanlama
    # bağlamıdır; karıştırmak iki sinyali de bulanıklaştırır.
    # B3-1: momentum bir GETİRİ ölçüsüdür → TOPLAM GETİRİ serisi
    m["_hist_tr"] = b.get("hist_tr")
    m["momentum"] = momentum_12_1(
        m["_hist_tr"].to_frame("Close")
        if m.get("_hist_tr") is not None else m["_hist"])
    m["pead"] = pead_sinyali(m.get("catalysts"))
    m["kademe"] = kademeli_plan(m)
    m["ufuk"] = ufuk_analizi(m, b.get("hist"), a)
    m["stop"] = stop_plani(m, a)
    # ÖZ-DENETİM: ekrana çıkacak her seviye önce sınanır (bkz.
    # plan_saglamasi). Sorun varsa arayüz planı GÖSTERMEZ.
    try:
        m["plan_saglama"] = plan_saglamasi(m) or None
    except Exception as _e:
        m["plan_saglama"] = [f"Sağlama çalıştırılamadı: {type(_e).__name__}"]
    try:
        # CANLI BULGU: eski hâli `(m.get("_hist") or pd.DataFrame())` idi.
        # pandas, DataFrame'e bool() uygulanmasını YASAKLAR → `or` her
        # seferinde ValueError atıyor, except dalına düşüyordu: MA200 ve
        # trend rejimi HİÇBİR ZAMAN hesaplanmadı (INTC paketinde
        # ma200_ustu=null bunun kanıtı).
        _h = m.get("_hist")
        _hc = _h["Close"] if (_h is not None and "Close" in _h) else None
        if _hc is not None and len(_hc.dropna()) >= 200:
            _ma200 = float(_hc.dropna().rolling(200).mean().iloc[-1])
            m["ma200"] = round(_ma200, 2)
            m["trend_ok"] = bool(m.get("price") and m["price"] >= _ma200)
        else:
            m["ma200"], m["trend_ok"] = None, None
    except Exception:
        m["ma200"], m["trend_ok"] = None, None
    # ── REJİM ÖZETİ (rapor G10): karar ekranı için tek satırlık bağlam.
    _rj = []
    if m.get("trend_ok") is False:
        _rj.append("fiyat MA200 ALTINDA (ayı rejimi bağlamı)")
    if (m.get("momentum") or {}).get("yon") == "NEGATİF":
        _rj.append("12-1 momentum NEGATİF")
    _pz = m.get("pead") or {}
    if _pz.get("yon") == "NEGATİF" and _pz.get("suruklenme_penceresinde"):
        _rj.append("negatif kazanç sürprizi sürüklenmesi (PEAD) aktif")
    m["rejim_uyari"] = _rj or None
    m["seviye_senaryo"] = None
    try:
        _svl = []
        _dz = ((m.get("sr") or {}).get("destekler") or [])
        _rz = ((m.get("sr") or {}).get("direncler") or [])
        if _dz:
            _svl.append(("En yakın destek altı", _dz[0]["bant_alt"]))
        if _rz:
            _svl.append(("En yakın direnç üstü", _rz[0]["bant_ust"]))
        # Aynı hata burada da vardı: istisna, DESTEK/DİRENÇ senaryoları
        # işlenmeden bloğu iptal ediyordu → seviye_senaryolari HER ZAMAN
        # null (INTC paketinde de null). Talimat 18-C bu alanı kullanıyor.
        _h2 = m.get("_hist")
        _hc = _h2["Close"] if (_h2 is not None and "Close" in _h2) else None
        if _hc is not None and len(_hc.dropna()) >= 200:
            _svl.append(("MA200",
                         float(_hc.dropna().rolling(200).mean().iloc[-1])))
        _out = []
        for _ad, _sv in _svl[:3]:
            _r = seviye_kosullu_istatistik(m["_hist"], _sv, m.get("atr22"),
                                           teyit=2)
            if _r:
                _r["ad"] = _ad
                _out.append(_r)
        m["seviye_senaryo"] = _out or None
    except Exception:
        m["seviye_senaryo"] = None
    m["trap"] = cyclical_trap(m)        # skor+kalite+revizyon sonrası
    # Aşama 2 — sektör merceği, sermaye tahsisi, baz oranlar, dosyalamalar
    m["_hist"] = b.get("hist")
    m["sector_tpl"] = sector_template(m)
    m["capalloc"] = capital_allocation(m)
    m["baserate"] = base_rates(m, a)
    # (PIOTROSKI kullanım sınırı bloğu YUKARI taşındı — bayrak, artık
    # kalite/skor/hüküm HESAPLANMADAN ÖNCE ekleniyor; denetim Bulgu 4.)
    m["filings"] = b.get("filings")
    _ed = b.get("edgar") or {}
    m["yabanci_dosyalayici"] = bool(_ed.get("yabanci_dosyalayici"))
    if m["yabanci_dosyalayici"]:
        m["adr_notu"] = (
            "Bu bir ADR / yabancı dosyalayıcı (20-F). Sonuçlar: (1) SEC'e "
            "10-Q DOSYALANMAZ → çeyreklik SEC akışı ve TTM yoktur, Yahoo "
            "yedeği kullanılır; (2) Form 4 muafiyeti var → içeriden işlem "
            "verisi ABD kapsamında YOKTUR ('alım yok' burada anlamsızdır); "
            "(3) finansallar us-gaap yerine ifrs-full etiketlidir → bazı "
            "çapraz doğrulamalar boş kalabilir. Bu boşluklar veri hatası "
            "değil, mevzuat farkıdır.")
        m["sanity"].append(("not", m["adr_notu"]))
    try:
        m["mutabakat"] = mutabakat_raporu(m, b)
    except Exception:
        m["mutabakat"] = None
    if b.get("hist_kaynak_notu"):
        m["sanity"].append(("uyarı", b["hist_kaynak_notu"]))
    m["tazelik"] = veri_tazeligi(m, b)
    if (m["tazelik"] or {}).get("uyari"):
        m["sanity"].append(("uyarı", m["tazelik"]["uyari"]))
    # Aşama 3 — normalize kazanç (SEC derinliği); fwd yukarıda kuruldu
    m["norm"] = normalized_earnings(m, a)
    if (m["fair"] is None and m["p_fcf"] is None and m["pe_t"] is None
            and m["ev_ebitda"] is None):
        # Elimizde tek bir değerleme sinyali bile yokken hüküm vermek yalan
        # söylemektir — dürüst çıktı: değerlendirilemez.
        m["verdict"] = ("VERİ YETERSİZ — HÜKÜM YOK", "⚪", "warning",
                        "yfinance bu ticker için değerleme sinyali üretecek "
                        "veri döndürmedi (FCF, F/K, EV/EBITDA hiçbiri yok). "
                        "Veri olmadan hüküm verilmez; ticker'ı doğrula ya da "
                        "birincil kaynağa (SEC EDGAR) in.")
        m["score"] = None    # skor da uydurulmaz — gösterge boş kalır
        m["kademe"] = None   # veri yokken alım planı üretmek yalan olur
        m["stop"] = None     # plansız stop da verilmez
    m["assumptions"] = a
    return m


# ---------------------------------------------------------------------------
# 6) KIRMIZI BAYRAK BATARYASI (kural motoru)
# ---------------------------------------------------------------------------

def beneish_m(inc, bal, cf):
    """
    BENEISH M-SCORE: 8 değişkenli muhasebe manipülasyon olasılık modeli
    (Beneish 1999). M > −1.78 → manipülasyon olasılığı istatistiksel eşiğin
    üzerinde. Kanıt DEĞİL, adli inceleme çağrısıdır. Eksik bileşen nötr
    (1.0) alınır ve raporlanır; çekirdek girdiler yoksa None döner.
    """
    rev = by_year(get_row(inc, ROWS["revenue"]))
    gp = by_year(get_row(inc, ROWS["gross"]))
    recv = by_year(get_row(bal, ROWS["receivables"]))
    ca = by_year(get_row(bal, ROWS["curr_assets"]))
    ta = by_year(get_row(bal, ROWS["total_assets"]))
    ppe = by_year(get_row(bal, ROWS["ppe"]))
    cl = by_year(get_row(bal, ROWS["curr_liab"]))
    ltd = by_year(get_row(bal, ROWS["lt_debt"]))
    sga = by_year(get_row(inc, ROWS["sga"]))
    dep = by_year(get_row(cf, ROWS["dep"]))
    ni = by_year(get_row(inc, ROWS["net_income"]))
    cfo = by_year(get_row(cf, ROWS["cfo"]))

    ys = sorted(set(rev) & set(ta))
    if len(ys) < 2:
        return None
    t, p = ys[-1], ys[-2]
    if not (rev.get(t) and rev.get(p) and ta.get(t) and ta.get(p)):
        return None
    if ni.get(t) is None or cfo.get(t) is None:
        return None                          # TATA çekirdektir

    eksik = []

    def ratio(num, den):
        return (num / den) if (num is not None and den) else None

    def index(cur, prev, name):
        if cur is None or prev in (None, 0):
            eksik.append(name)
            return 1.0
        return cur / prev

    dsri = index(ratio(recv.get(t), rev[t]), ratio(recv.get(p), rev[p]), "DSRI")
    gmi = index(ratio(gp.get(p), rev[p]), ratio(gp.get(t), rev[t]), "GMI")
    qa_t = (1 - (ca.get(t, 0) + ppe.get(t, 0)) / ta[t]) if ta.get(t) else None
    qa_p = (1 - (ca.get(p, 0) + ppe.get(p, 0)) / ta[p]) if ta.get(p) else None
    if qa_t is not None and qa_p not in (None, 0) and qa_p > 0 and qa_t > 0:
        aqi = qa_t / qa_p
    else:
        eksik.append("AQI"); aqi = 1.0
    sgi = rev[t] / rev[p]
    dr_t = ratio(dep.get(t), (dep.get(t) or 0) + (ppe.get(t) or 0))
    dr_p = ratio(dep.get(p), (dep.get(p) or 0) + (ppe.get(p) or 0))
    depi = index(dr_p, dr_t, "DEPI")
    sgai = index(ratio(sga.get(t), rev[t]), ratio(sga.get(p), rev[p]), "SGAI")
    lev_t = ratio((cl.get(t, 0) + ltd.get(t, 0)), ta[t])
    lev_p = ratio((cl.get(p, 0) + ltd.get(p, 0)), ta[p])
    lvgi = index(lev_t, lev_p, "LVGI")
    tata = (ni[t] - cfo[t]) / ta[t]

    # KATSAYI DOĞRULAMASI (bütünsel rapor G3): TATA katsayısı 4.679'dur
    # (Beneish 1999); bazı ikincil kaynaklardaki 4.697 HATALIDIR. Diğer 7
    # katsayı ve eşik etiketi (−2.22 / −1.78) kanonikle birebir doğrulandı.
    m_score = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
               + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    bolge = ("RİSKLİ" if m_score > -1.78
             else "GRİ" if m_score > -2.22 else "TEMİZ")
    return {"skor": round(m_score, 3), "bolge": bolge,
            "esikler": "TEMİZ < −2.22 < GRİ < −1.78 < RİSKLİ",
            "donem": f"{p}→{t}",
            "bilesenler": {"DSRI": round(dsri, 3), "GMI": round(gmi, 3),
                           "AQI": round(aqi, 3), "SGI": round(sgi, 3),
                           "DEPI": round(depi, 3), "SGAI": round(sgai, 3),
                           "TATA": round(tata, 4), "LVGI": round(lvgi, 3)},
            "eksik_notr_alinan": eksik or None}


def parse_catalysts(b):
    """Yahoo takviminden yaklaşan olaylar + kazanç sürpriz geçmişi."""
    out = {"gelecek": [], "kazanc_surpriz_gecmisi": [],
           "not": "Yalnızca kazanç/temettü takvimi; sözleşme yenilemesi, "
                  "regülasyon, dava gibi şirkete özel olayları Claude web "
                  "katmanı arar."}
    today = pd.Timestamp.now().normalize()

    def add(ev, dt):
        try:
            ts = pd.Timestamp(dt)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
        except Exception:
            return
        gk = int((ts.normalize() - today).days)
        if -1 <= gk <= 180:
            out["gelecek"].append({"olay": ev, "tarih": str(ts.date()),
                                   "gun_kaldi": gk})

    cal = b.get("calendar") or {}
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date")
        for d in (ed if isinstance(ed, (list, tuple)) else [ed] if ed else []):
            add("Kazanç açıklaması (tahmini)", d)
        if cal.get("Ex-Dividend Date"):
            add("Temettü hak düşümü (ex-div)", cal["Ex-Dividend Date"])
        if cal.get("Dividend Date"):
            add("Temettü ödemesi", cal["Dividend Date"])

    eds = b.get("earnings_dates")
    if eds is not None and not getattr(eds, "empty", True):
        for idx_, row in eds.iterrows():
            try:
                ts = pd.Timestamp(idx_)
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
            except Exception:
                continue
            rep = _num(row.get("Reported EPS"))
            if rep is not None:
                out["kazanc_surpriz_gecmisi"].append(
                    {"tarih": str(ts.date()),
                     "tahmin": _num(row.get("EPS Estimate")),
                     "gerceklesen": rep,
                     "surpriz": _num(row.get("Surprise(%)"))})
            elif ts > pd.Timestamp.now():
                add("Kazanç açıklaması", ts)
    # DENETİM BULGUSU W: liste SIRALANMADAN [:4] alınıyordu. yfinance
    # genelde yeniden eskiye verir ama GARANTİ değildir — kaynak eskiden
    # yeniye verirse "son 4 çeyrek" başlığı altında EN ESKİ 4 çeyrek
    # gösteriliyordu (testte doğrulandı). Artık tarihe göre azalan sıralanıp
    # alınıyor, sonra okuma kolaylığı için yeniden artan sıraya konuyor.
    out["kazanc_surpriz_gecmisi"] = sorted(
        sorted(out["kazanc_surpriz_gecmisi"],
               key=lambda x: x["tarih"], reverse=True)[:4],
        key=lambda x: x["tarih"])

    # Son 8-K kritik olayları "geçmiş materyal olay" olarak ekle (tez-bozan
    # sinyal olabilir; takvim değil ama karar için önemli).
    o8 = b.get("olaylar_8k") or {}
    kb = o8.get("kirmizi_bayrak") or []
    out["materyal_8k"] = [
        {"tarih": o["tarih"],
         "ozet": ", ".join(f"{it['kod']} {it['aciklama']}"
                           for it in o["itemler"][:2]),
         "onem": o["en_yuksek_onem"], "link": o["link"]}
        for o in (o8.get("olaylar") or [])
        if o["en_yuksek_onem"] in ("kritik", "yuksek")][:5]
    if kb:
        out["kirmizi_bayrak_8k"] = len(kb)

    seen, ded = set(), []
    for e in sorted(out["gelecek"], key=lambda x: x["gun_kaldi"]):
        k = ("Kazanç" if "Kazanç" in e["olay"] else e["olay"], e["tarih"])
        if k not in seen:
            seen.add(k); ded.append(e)
    out["gelecek"] = ded[:6]
    return out


def _insider_sec(form4, sec_link):
    """SEC Form 4 dosyalarından içeriden-işlem özeti (BİRİNCİL kaynak)."""
    rows = []
    for f4 in form4.get("dosyalar") or []:
        for i in f4.get("islemler") or []:
            try:
                ts = pd.Timestamp(i.get("tarih") or f4.get("dosya_tarihi"))
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
            except Exception:
                ts = None
            rows.append({"_ts": ts, "kisi": (f4.get("kisi") or "")[:40],
                         "unvan": (f4.get("unvan") or "")[:40],
                         "tur": i.get("tur"), "adet": i.get("adet"),
                         "deger_usd": i.get("deger_usd"),
                         "planli": bool(f4.get("planli_10b5_1")),
                         "aciklama": ((i.get("kod") or "") +
                                      (" · 10b5-1 planlı"
                                       if f4.get("planli_10b5_1") else ""))})
    if not rows:
        return None
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    win = [r for r in rows if r["_ts"] is None or r["_ts"] >= cutoff]
    alim = [r for r in win if r["tur"] == "ALIM"]
    satim = [r for r in win if r["tur"] == "SATIM"]
    planli_satis = sum(1 for r in satim if r.get("planli"))
    kume = None
    buys = sorted([r for r in alim if r["_ts"] is not None],
                  key=lambda r: r["_ts"])
    if buys:
        kume = False
        for r0 in buys:
            kisiler = {r["kisi"] for r in buys
                       if 0 <= (r["_ts"] - r0["_ts"]).days <= 30}
            if len(kisiler) >= 3:
                kume = True
                break

    def _toplam(lst):
        v = [x["deger_usd"] for x in lst if x.get("deger_usd")]
        return float(sum(v)) if v else None

    son = sorted([r for r in rows if r["_ts"] is not None],
                 key=lambda r: r["_ts"], reverse=True)[:8] or rows[:8]
    son_out = [{"tarih": (str(r["_ts"].date()) if r["_ts"] is not None
                          else None),
                "kisi": r["kisi"], "unvan": r["unvan"], "tur": r["tur"],
                "adet": r["adet"], "deger_usd": r["deger_usd"],
                "aciklama": r["aciklama"]} for r in son]
    return {"ozet": {"alim_adet": len(alim), "satim_adet": len(satim),
                     "alim_usd": _toplam(alim), "satim_usd": _toplam(satim),
                     "kume_alimi_30g_3insider": kume,
                     "planli_satis_adet": planli_satis,
                     "kaynak": "SEC Form 4 (birincil)",
                     "pencere": "son 12 ay"},
            "son_islemler": son_out,
            "sec_link": sec_link or form4.get("link"),
            "not": ("Kaynak: SEC Form 4 dosyalamalarının KENDİSİ (kod "
                    "P=açık piyasa alımı, S=satış, M=opsiyon, A=ödül). "
                    "10b5-1 işaretli satışlar ÖNCEDEN PLANLIDIR — sinyal "
                    "gücü düşüktür. Açık piyasa alımı çoğu şirkette "
                    "NADİRDİR; 'alım yok' tek başına ayı sinyali değildir.")}


def parse_insider(df, form4=None, sec_link=None):
    """
    İçeriden işlemler — iki katmanlı kaynak stratejisi:
      1) SEC Form 4 (BİRİNCİL): dosyalamanın kendisinden P/S/M/A kodları ve
         10b5-1 planlı-satış bayrağı. 'Alım görünmüyor' burada GERÇEK bir
         yokluktur.
      2) Yahoo özeti (YEDEK): SEC erişilemezse. Seyrek/gecikmeli olabilir
         ve 10b5-1 ayrımı taşımaz — dürüstçe etiketlenir.
    """
    if form4 and form4.get("durum") == "OK":
        sec = _insider_sec(form4, sec_link)
        if sec:
            return sec
    if form4 and form4.get("durum") == "BOŞ":
        return {"ozet": {"alim_adet": 0, "satim_adet": 0, "alim_usd": None,
                         "satim_usd": None,
                         "kume_alimi_30g_3insider": False,
                         "planli_satis_adet": 0,
                         "kaynak": "SEC Form 4 (birincil)",
                         "pencere": "son dosyalamalar"},
                "son_islemler": None,
                "sec_link": sec_link or form4.get("link"),
                "not": ("SEC'te yakın dönemde Form 4 dosyalaması "
                        "GÖRÜNMÜYOR — kaynak birincil olduğu için bu "
                        "gerçek bir sessizliktir, veri eksiği değil.")}
    if df is None or getattr(df, "empty", True):
        return None
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    col = {c.lower(): c for c in d.columns}
    c_txt = col.get("text") or col.get("transaction")
    c_ins = col.get("insider") or col.get("name")
    c_pos = col.get("position")
    c_val = col.get("value")
    c_sh = col.get("shares")
    c_dt = next((v for k, v in col.items() if "date" in k), None)

    rows = []
    for _, r in d.iterrows():
        txt = str(r.get(c_txt, "") or "") if c_txt else ""
        low = txt.lower()
        if "purchase" in low or low.startswith("buy"):
            tur = "ALIM"
        elif "sale" in low or "sold" in low:
            tur = "SATIM"
        else:
            tur = "DİĞER"
        ts = None
        if c_dt is not None:
            try:
                ts = pd.Timestamp(r[c_dt])
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
            except Exception:
                ts = None
        rows.append({"_ts": ts, "kisi": (str(r.get(c_ins, ""))[:40]
                                         if c_ins else ""),
                     "unvan": str(r.get(c_pos, ""))[:40] if c_pos else "",
                     "tur": tur,
                     "adet": _num(r.get(c_sh)) if c_sh else None,
                     "deger_usd": _num(r.get(c_val)) if c_val else None,
                     "aciklama": txt[:60]})

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
    win = [r for r in rows if r["_ts"] is None or r["_ts"] >= cutoff]
    alim = [r for r in win if r["tur"] == "ALIM"]
    satim = [r for r in win if r["tur"] == "SATIM"]

    kume = None
    buys = sorted([r for r in alim if r["_ts"] is not None],
                  key=lambda r: r["_ts"])
    if buys:
        kume = False
        for r0 in buys:
            kisiler = {r["kisi"] for r in buys
                       if 0 <= (r["_ts"] - r0["_ts"]).days <= 30}
            if len(kisiler) >= 3:
                kume = True
                break

    def toplam(lst):
        v = [x["deger_usd"] for x in lst if x.get("deger_usd")]
        return float(sum(v)) if v else None

    son = sorted([r for r in rows if r["_ts"] is not None],
                 key=lambda r: r["_ts"], reverse=True)[:8] or rows[:8]
    son_out = [{"tarih": (str(r["_ts"].date()) if r["_ts"] is not None
                          else None),
                "kisi": r["kisi"], "unvan": r["unvan"], "tur": r["tur"],
                "adet": r["adet"], "deger_usd": r["deger_usd"],
                "aciklama": r["aciklama"]} for r in son]
    return {"ozet": {"alim_adet": len(alim), "satim_adet": len(satim),
                     "alim_usd": toplam(alim), "satim_usd": toplam(satim),
                     "kume_alimi_30g_3insider": kume,
                     "planli_satis_adet": None,
                     "kaynak": "Yahoo (yedek)",
                     "pencere": "son 12 ay (tarih yoksa tümü)"},
            "son_islemler": son_out,
            "sec_link": sec_link,
            "not": ("Kaynak: Yahoo Form 4 ÖZETİ (yedek) — SEYREK ve "
                    "gecikmeli olabilir; 10b5-1 planlı-satış ayrımı yoktur. "
                    "'Alım görünmüyor' burada iki anlama gelebilir: gerçekten "
                    "alım yok (çoğu şirkette normaldir, alımlar nadirdir) YA "
                    "DA Yahoo o satırları taşımıyor. Kesin liste için SEC "
                    "Form 4 linkine bak; yokluk tek başına sinyal değildir.")}


def parse_revisions(et, er, pt, recs, price):
    """Analist tahmin revizyon momentumu + hedef fiyat + öneri dağılımı."""
    out = {}

    def row(df, key):
        try:
            if df is None or getattr(df, "empty", True):
                return None
            idx = [str(i) for i in df.index]
            if key not in idx:
                return None
            return df.iloc[idx.index(key)]
        except Exception:
            return None

    for key, name in (("0y", "eps_bu_yil"), ("+1y", "eps_gelecek_yil")):
        r = row(et, key)
        if r is not None:
            cols = {str(c).lower(): c for c in r.index}
            cur = _num(r.get(cols.get("current")))
            old = _num(r.get(cols.get("90daysago")))
            chg = ((cur / old - 1)
                   if (cur is not None and old not in (None, 0)) else None)
            out[name] = {"simdi": cur, "90g_once": old, "degisim": chg}

    try:
        if er is not None and not getattr(er, "empty", True):
            cols = {str(c).lower(): c for c in er.columns}
            u, dcol = cols.get("uplast30days"), cols.get("downlast30days")
            ups = (int(pd.to_numeric(er[u], errors="coerce").fillna(0).sum())
                   if u else None)
            downs = (int(pd.to_numeric(er[dcol], errors="coerce")
                         .fillna(0).sum()) if dcol else None)
            if ups is not None or downs is not None:
                out["yukari_30g"] = ups or 0
                out["asagi_30g"] = downs or 0
                out["skor_net_30g"] = (ups or 0) - (downs or 0)
    except Exception:
        pass

    try:
        if isinstance(pt, dict) and pt:
            mean = _num(pt.get("mean"))
            out["hedef_fiyat"] = {
                "ortalama": mean, "medyan": _num(pt.get("median")),
                "dusuk": _num(pt.get("low")), "yuksek": _num(pt.get("high")),
                "potansiyel": ((mean / price - 1)
                               if (mean and price) else None)}
    except Exception:
        pass

    try:
        if recs is not None and not getattr(recs, "empty", True):
            r0 = recs.iloc[0]
            keys = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            out["oneri_dagilimi"] = {k: int(r0[k]) for k in keys
                                     if k in recs.columns}
    except Exception:
        pass

    chg = (out.get("eps_bu_yil") or {}).get("degisim")
    net = out.get("skor_net_30g")
    if net is None and chg is None:
        return out or None
    # ── B6-7 ─────────────────────────────────────────────────────────
    # Yön kararı HAM NET SAYIYA bakıyordu; ham sayı ANALİST KAPSAMASINA
    # bağımlıdır. ÖLÇÜM: 6 analistli hissede net −3 (oran −0.60, felaket)
    # kod NÖTR diyor; 30 analistli hissede net −5 (oran −0.17, hafif) kod
    # ALEYHTE diyor — hem kaçırıyor hem yanlış alarm veriyor.
    _u, _d = out.get("yukari_30g"), out.get("asagi_30g")
    _oran = None
    if _u is not None and _d is not None and (_u + _d) > 0:
        _oran = (_u - _d) / (_u + _d)
        out["revizyon_orani"] = round(_oran, 3)
        out["analist_hareketi"] = int(_u + _d)
    down = ((_oran is not None and _oran <= -0.20)
            or (chg is not None and chg < -0.03))
    up = ((_oran is not None and _oran >= 0.20)
          or (chg is not None and chg > 0.03))
    out["yon"] = "AŞAĞI" if down and not up else "YUKARI" if up else "YATAY"
    out["yon_notu"] = (
        "Yön, HAM SAYI değil REVİZYON ORANI ile belirlenir: "
        "(yukarı−aşağı)/(yukarı+aşağı). Ham net sayı analist kapsamasına "
        "bağımlıdır. Revizyon momentumu gerçektir (Chan-Jegadeesh-"
        "Lakonishok 1996; sürüklenme ~3-6 ay) ama yayın sonrası ~%50 "
        "zayıflamıştır (McLean-Pontiff 2016).")
    return out


def sanity_checks(m):
    """
    SESSİZ DENETÇİ: kritik rakamları iki bağımsız yoldan hesaplayıp
    karşılaştırır. Amaç, veri kaynağı (Yahoo) hatalarını rapora sızmadan
    yakalamak. Bunlar ŞİRKET bayrağı DEĞİL, VERİ KALİTESİ notudur — skoru
    etkilemez; kullanıcıya ve Claude'a şeffafça sunulur.
    """
    notes = []
    _b = lambda v: (f"{v/1e9:,.2f}B" if abs(v) >= 1e9 else f"{v/1e6:,.1f}M")

    # 1) Piyasa değeri çaprazı: fiyat×hisse vs Yahoo'nun hazır değeri
    mc_calc = ((m.get("price") or 0) * (m.get("shares") or 0)) or None
    mc_info = m.get("mcap_info")
    if mc_calc and mc_info and abs(mc_calc - mc_info) / mc_info > 0.03:
        notes.append(("uyarı",
                      f"Piyasa değeri iki yoldan uyuşmuyor: fiyat×hisse "
                      f"${_b(mc_calc)} vs Yahoo ${_b(mc_info)} (fark "
                      f"%{abs(mc_calc - mc_info) / mc_info * 100:.1f}) — "
                      f"hisse sayısı veya fiyat bayat olabilir."))

    # 2) 52 hafta çaprazı: hesapladığımız vs Yahoo'nun söylediği
    for et, ours, yh in (("zirvesi", m.get("hi52"), m.get("hi52_info")),
                         ("dibi", m.get("lo52"), m.get("lo52_info"))):
        if ours and yh and abs(ours - yh) / max(ours, yh) > 0.03:
            notes.append(("uyarı",
                          f"52 hafta {et} uyuşmuyor: geçmiş veriden "
                          f"{ours:,.2f}, Yahoo {yh:,.2f} — geçmiş veriden "
                          f"hesaplanan kullanıldı (Yahoo'nun bu alanı hatalı "
                          f"olabiliyor)."))

    # 3) ROIC ≠ ROE yapısal kontrolü
    r_ic, r_oe = m.get("roic"), m.get("roe")
    if r_ic is not None and r_oe is not None and abs(r_ic - r_oe) < 1e-9:
        notes.append(("uyarı", "ROIC ile ROE birebir aynı çıktı — hesap "
                               "zinciri şüpheli, ROIC'e güvenme."))

    # 4) EPS çaprazı: net kâr / ort. seyreltilmiş hisse vs Yahoo trailingEps
    dsh = m.get("dil_shares_seri") or {}
    ni0 = latest(m.get("ni") or {})
    eps_i = m.get("eps_info")
    if dsh and ni0 is not None and eps_i:
        d0 = dsh[sorted(dsh)[-1]]
        if d0:
            eps_c = ni0 / d0
            if abs(eps_c - eps_i) / max(abs(eps_i), 0.01) > 0.20:
                notes.append(("not",
                              f"EPS çaprazı sapıyor: hesap {eps_c:.2f} vs "
                              f"Yahoo {eps_i:.2f} — TTM/yıllık dönem farkı "
                              f"veya hisse serisi tutarsızlığı olabilir."))

    # 5) TTM FCF vs son yıllık FCF makası (tek seferlik kalem dedektörü)
    t, yy = m.get("ttm_fcf_raw"), latest(m.get("fcf") or {})
    if t and yy and yy != 0:
        r = t / yy
        if r < 0.4 or r > 2.5:
            notes.append(("not",
                          f"TTM FCF (${_b(t)}) ile son yıllık FCF (${_b(yy)}) "
                          f"arasında {r:.1f}x makas — tek seferlik kalem veya "
                          f"mevsimsellik olabilir; DCF bazını sorgula."))

    # 6) IPO/dönüşüm sıçraması bilgilendirmesi
    if m.get("dilution_ipo_jump"):
        notes.append(("not",
                      f"Hisse sayısında tek seferlik sıçrama tespit edildi "
                      f"({m['dilution_ipo_jump']}) — IPO/tercihli hisse "
                      f"dönüşümü olabilir; SÜREKLİ dilüsyon SAYILMADI, temiz "
                      f"yıllık medyan kullanıldı."))

    # 7) Negatif özkaynak
    if m.get("equity") is not None and m["equity"] < 0:
        notes.append(("not", "Özkaynak negatif — ROE ve P/B bu şirkette "
                             "anlamsızdır (ROE bilinçli olarak boş "
                             "bırakıldı)."))
    # 7b) Borç verisi eksik → net borç hesaplanmadı (0 VARSAYILMADI)
    if m.get("total_debt") is None:
        notes.append(("not", "Toplam borç verisi alınamadı — net borç, "
                             "NB/EBITDA ve DCF'teki net borç düzeltmesi "
                             "hesaplanamadı; borç 0 VARSAYILMADI, ilgili "
                             "alanlar boş bırakıldı."))
    # 8) SEC EDGAR çaprazı: resmî 10-K ile >%10 sapan kalemler
    ed = m.get("edgar") or {}
    for p in ed.get("karsilastirma") or []:
        if p["uyum"] == "✗":
            notes.append(("uyarı",
                          f"SEC uyuşmazlığı — {p['kalem']}: Yahoo "
                          f"{p['yahoo']:,.0f} vs SEC {p['sec']:,.0f} "
                          f"({p['sec_yil']}) — sapma %{p['sapma_pct']}. "
                          f"Resmî değer SEC'tir; Yahoo kalemini [HİPOTEZ] "
                          f"say."))
    return notes


def forensic_flags(m):
    """
    GENİŞLETİLMİŞ ADLİ BAYRAKLAR (Schilit 'Financial Shenanigans' + Apostolou
    eşikleri). Beneish'e EK olarak somut nakit/tahakkuk kontrolleri. Bunlar
    katı kesim değil DİKKAT bayrağıdır — biriktikçe kâr kalitesi şüphesi
    artar. Şirket bayrağı olarak F listesine eklenir.
    """
    F = []

    def flag(sev, txt):
        F.append((sev, txt))

    inc, bal, cf = m.get("_inc"), m.get("_bal"), m.get("_cf")
    if inc is None or cf is None:
        return F
    rev = by_year(get_row(inc, ROWS["revenue"]))
    ni = by_year(get_row(inc, ROWS["net_income"]))
    cfo = by_year(get_row(cf, ROWS["cfo"]))
    recv = by_year(get_row(bal, ROWS["receivables"])) if bal is not None else {}
    inv = by_year(get_row(bal, ROWS["inventory"])) if bal is not None else {}
    cogs = by_year(get_row(inc, ROWS["cogs"]))

    # 1) Nakit-kâr makası: OCF/NI birden çok yıl < 0.8 → kâr nakde dönmüyor
    ys = sorted(set(ni) & set(cfo))
    if len(ys) >= 2:
        oran = [(y, cfo[y] / ni[y]) for y in ys[-3:]
                if ni.get(y) and ni[y] > 0]
        dusuk = [(y, r) for y, r in oran if r < 0.8]
        if len(dusuk) >= 2:
            son_y, son_r = dusuk[-1]
            flag("yüksek", f"Nakit-kâr makası: son {len(dusuk)} yılda "
                           f"işletme nakdi/net kâr < 0.8 (son: {son_r:.2f}) "
                           f"— raporlanan kâr nakde tam dönüşmüyor, kâr "
                           f"kalitesi şüpheli.")

    # 2) Alacak gelirin %25'ini aşıyor (dikkat eşiği; kod ve mesaj %25)
    if rev and recv:
        y = max(set(rev) & set(recv), default=None)
        if y and rev.get(y) and recv.get(y):
            r = recv[y] / rev[y]
            # B5-5: Schilit'in (Financial Shenanigans) yayımlanmış kuralı
            # alacakların yıllık satışın ~%15'ini aşmasıdır. Kod %25
            # kullanıyordu — kaynaktan GEVŞEK; gelirin %16-25'i arasındaki
            # yığılmayı (Schilit'in bayrak bastığı bölge) hiç görmüyordu.
            if r > 0.15:
                flag("yüksek" if r > 0.25 else "orta",
                     f"Alacaklar gelirin %{r*100:.0f}'i (Schilit dikkat "
                     f"eşiği %15) — tahsilat/gelir tanıma incelenmeli."
                     + (" Kanal doldurma ya da agresif gelir tanımanın "
                        "klasik izidir." if r > 0.25 else ""))

    # 3) Stok > COGS'un %25'i + stok satıştan hızlı büyüyor
    if cogs and inv:
        y = max(set(cogs) & set(inv), default=None)
        if y and cogs.get(y) and inv.get(y):
            r = inv[y] / cogs[y]
            iy = sorted(set(inv) & set(rev))
            hizli = (len(iy) >= 2 and inv[iy[-1]] and inv[iy[-2]] and
                     rev.get(iy[-1]) and rev.get(iy[-2]) and
                     (inv[iy[-1]] / inv[iy[-2]] - 1) >
                     (rev[iy[-1]] / rev[iy[-2]] - 1) + 0.10)
            if r > 0.25 and hizli:
                flag("orta", f"Stok COGS'un %{r*100:.0f}'i ve satıştan hızlı "
                             f"büyüyor — talep yavaşlaması / eskime riski.")
    return F


def build_red_flags(m):
    F = []

    def flag(sev, txt):
        F.append((sev, txt))

    if m["fcf_base"] is None or m["fcf_base"] <= 0:
        flag("yüksek", "Serbest nakit akışı (baz) sıfır/negatif — DCF anlamsız; "
                       "fiyatlama tamamen gelecek vaadine yaslanıyor.")
    if m.get("cfo_lt_ni_2y"):
        flag("yüksek", "Son 2 yıldır CFO < Net Kâr — kârın nakit karşılığı zayıf "
                       "(tahakkuk kalitesi sorunu).")
    ac = m.get("accrual")
    if ac is not None and ac > 0.05:
        flag("orta", f"Tahakkuk oranı yüksek ({fmt_pct(ac)}) — Sloan anomalisi "
                     f"bölgesi, kâr kalitesini sorgula.")
    nd = m.get("nd_ebitda")
    if nd is not None:
        if nd > 4:
            flag("yüksek", f"Net Borç/EBITDA {fmt_x(nd)} — ağır kaldıraç, "
                           f"faiz ortamına ve krize kırılgan.")
        elif nd > 3:
            flag("orta", f"Net Borç/EBITDA {fmt_x(nd)} — kaldıraç izlenmeli.")
    ic = m.get("int_cov")
    if ic is not None:
        if ic < 1.5:
            flag("yüksek", f"Faiz karşılama {fmt_x(ic)} — EBIT faizi zor "
                           f"karşılıyor, borç çevirme riski.")
        elif ic < 3:
            flag("orta", f"Faiz karşılama {fmt_x(ic)} — konfor alanı dar.")
    cr = m.get("current_ratio")
    if cr is not None and cr < 1:
        flag("orta", f"Cari oran {fmt_f(cr)} < 1 — kısa vadeli likidite gergin.")
    dil = m.get("dilution_yoy")              # temiz yıllık medyan (IPO
    if dil is not None:                      # sıçraması hariç tutulmuş)
        if dil > 0.06:
            flag("yüksek", f"Sürekli dilüsyon: ortalama seyreltilmiş hisse "
                           f"yılda {fmt_pct(dil)} artıyor — hissedar payı "
                           f"eriyor.")
        elif dil > 0.03:
            flag("orta", f"Ortalama seyreltilmiş hisse yılda {fmt_pct(dil)} "
                         f"artıyor (dilüsyon).")
    sr = m.get("sbc_rev")
    if sr is not None and sr > 0.10:
        flag("orta", f"SBC/Gelir {fmt_pct(sr)} — hisse bazlı tazminat gerçek bir "
                     f"maliyet; 'düzeltilmiş' kârlara temkinli yaklaş.")
    dso = m.get("dso") or {}
    if len(dso) >= 2:
        ys = sorted(dso.keys())
        if dso[ys[-1]] > dso[ys[-2]] * 1.20:
            flag("orta", f"DSO {dso[ys[-2]]:.0f}→{dso[ys[-1]]:.0f} gün — "
                         f"alacaklar gelirden hızlı büyüyor (kanal doldurma / "
                         f"tahsilat sorunu ihtimali).")
    gm = m.get("gm") or {}
    if len(gm) >= 3:
        ys = sorted(gm.keys())
        if gm[ys[-1]] < gm[ys[-3]] - 0.03:
            flag("orta", f"Brüt marj eriyor ({fmt_pct(gm[ys[-3]])} → "
                         f"{fmt_pct(gm[ys[-1]])}) — fiyatlama gücü zayıflıyor "
                         f"olabilir.")
    z = m.get("altman")
    # ── KANIT TURU (araştırma C1): Altman'ın X4 bileşeni = piyasa değeri /
    # toplam yükümlülük. Piyasa değeri şişkin bir şirkette X4 tek başına
    # skoru "güvenli"ye taşır — canlı örnekte faaliyet marjı -%4 ve ROIC
    # negatifken Z=3.25 "güvenli" çıktı. Campbell-Hilscher-Szilagyi (2008)
    # dinamik modellerin Z'yi geçtiğini gösteriyor. Skoru DEĞİŞTİRMİYORUZ;
    # çelişki görünür kılınıyor.
    _abz0 = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                             ALTMAN_BOLGE["uretim"])
    # DİKKAT: doğru anahtar m["om"] (yıl→faaliyet marjı). İlk yazımda
    # var olmayan "op_margin_seri" kullanılmıştı — bayrak sessizce ölü
    # kalırdı; anahtar canlı payloadla doğrulandı.
    _op_m = (m.get("om") or {})
    _son_op = (_op_m[max(_op_m)] if _op_m else None)
    if (z is not None and z >= _abz0["guvenli"]
            and ((m.get("roic") is not None and m["roic"] < 0)
                 or (_son_op is not None and _son_op < 0))):
        m["altman_x4_uyarisi"] = (
            "Altman Z 'güvenli' diyor AMA faaliyet marjı/ROIC negatif. Z'nin "
            "X4 bileşeni piyasa değeri/borç oranıdır: yüksek piyasa değeri "
            "skoru tek başına güvenli bölgeye taşıyabilir. Yani BU SKORU "
            "YUKARI ÇEKEN ŞEY FİYATIN KENDİSİ olabilir — pahalılık skoruyla "
            "birlikte okunmalı. [Campbell-Hilscher-Szilagyi 2008]")
        flag("orta", m["altman_x4_uyarisi"])
    if z is not None:
        # TUR4 bulgusu: eşikler MODELİN KENDİ eşikleri (Z 1.81/2.99,
        # Z'' 1.10/2.60) — sabit klasik eşik, hizmet/teknoloji şirketine
        # kendi GÜVENLİ bölgesinde haksız bayrak basıyordu.
        _abf = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                                ALTMAN_BOLGE["uretim"])
        if z < _abf["tehlike"]:
            flag("yüksek", f"Altman Z {fmt_f(z)} < {_abf['tehlike']} — "
                           f"istatistiksel iflas riski bölgesi "
                           f"(model: {m.get('altman_model')}).")
        elif z < _abf["guvenli"]:
            flag("orta", f"Altman Z {fmt_f(z)} — gri bölge (güvenli eşik "
                         f"{_abf['guvenli']}, model "
                         f"{m.get('altman_model')}).")
    p = m.get("piotroski") or {}
    if p.get("max", 0) >= 6 and p.get("score", 9) <= 3:
        flag("orta", f"Piotroski F {p['score']}/{p['max']} — temel sağlık zayıf.")
    if m.get("ccy_mismatch"):
        flag("orta", f"Fiyat para birimi ({m['ccy']}) ile finansal tablo para "
                     f"birimi ({m['fin_ccy']}) farklı — hisse başı DCF'i kur "
                     f"düzeltmesi olmadan yorumlama.")
    pv = m.get("pv_gap")
    if pv is not None and pv > 3 and (m.get("price_chg_2y") or 0) > 0.5:
        flag("yüksek", f"Fiyat-değer makası: 2 yılda fiyat "
                       f"{fmt_pct(m['price_chg_2y'])} koşarken gelir "
                       f"{fmt_pct(m['rev_chg_2y'])} büyüdü (~{pv:.1f}x makas) "
                       f"— fiyat, temelin çok önünde bir hikâye fiyatlıyor.")
    band = m.get("ps_band") or {}
    pos_b = band.get("konum_0dip_1zirve")
    if pos_b is not None and pos_b > 0.90:
        konum_txt = (f"kendi tarihsel bandının ÜZERİNE taşmış "
                     f"(bandın tavanının ~{pos_b:.1f} katı yukarısında)"
                     if pos_b > 1.0 else
                     f"kendi tarihsel bandının zirvesinde (konum {pos_b:.0%})")
        flag("orta", f"F/S çarpanı {konum_txt} — bugüne kadarki getirinin bir "
                     f"kısmını çarpan genişlemesi taşımış; alım yeri "
                     f"açısından aleyhte.")
    rr, srs = m.get("rnd_rev") or {}, m.get("sbc_rev_seri") or {}
    ys_c = sorted(set(rr) & set(srs))
    if len(ys_c) >= 3:
        if (rr[ys_c[-1]] < rr[ys_c[-3]] - 0.01
                and srs[ys_c[-1]] > srs[ys_c[-3]] + 0.01):
            flag("yüksek", f"Söylem-gerçek çelişkisi: Ar-Ge/Gelir düşerken "
                           f"({fmt_pct(rr[ys_c[-3]])}→{fmt_pct(rr[ys_c[-1]])}) "
                           f"SBC/Gelir yükseliyor ({fmt_pct(srs[ys_c[-3]])}→"
                           f"{fmt_pct(srs[ys_c[-1]])}) — büyüme hikâyesi "
                           f"içeride finanse edilmiyor, yönetim primleniyor.")
    bm = m.get("beneish") or {}
    if bm.get("skor") is not None and bm["skor"] > -1.78:
        flag("yüksek", f"Beneish M-Skoru {bm['skor']:.2f} > −1.78 "
                       f"({bm.get('donem', '')}) — kâr manipülasyonu "
                       f"OLASILIĞI istatistiksel eşiğin üzerinde; bileşenler "
                       f"adli inceleme istiyor. (Kanıt değil, olasılık.)")
    rv = m.get("revizyon") or {}
    if rv.get("yon") == "AŞAĞI" and (rv.get("skor_net_30g") or 0) <= -5:
        flag("orta", f"Analist momentumu aleyhte: son 30 günde net "
                     f"{rv['skor_net_30g']} aşağı revizyon — tahminler "
                     f"kesiliyor.")
    ins_o = (m.get("insider") or {}).get("ozet") or {}
    if (ins_o.get("satim_adet") or 0) >= 5 and (ins_o.get("alim_adet") or 0) == 0:
        s_usd = ins_o.get("satim_usd")
        flag("orta", f"İçeriden akış tek yönlü: son 12 ayda "
                     f"{ins_o['satim_adet']} satış, SIFIR açık piyasa alımı"
                     + (f" (~${s_usd/1e6:.1f}M)." if s_usd else "."))
    ga = m.get("geri_alim") or {}
    if (ga.get("telafi_payi") or 0) >= 0.90 and (ga.get("geri_alim_musd") or 0) > 0:
        flag("orta", f"Geri alımın %{ga['telafi_payi']*100:.0f}'i SBC "
                     f"seyreltmesini telafi ediyor (${ga['sbc_telafisi_musd']:,.0f}M) "
                     f"— gerçek hissedar iadesi yalnız "
                     f"${ga['gercek_iade_musd']:,.0f}M. 'Geri alım yapıyor' "
                     f"görüntüsü yanıltıcı; bu tazminatın nakit maliyetidir.")
    elif ga.get("telafi_payi") is None and (ga.get("sbc_musd") or 0) > 0:
        flag("orta", f"SBC ${ga['sbc_musd']:,.0f}M ama geri alım YOK — "
                     f"seyreltme telafi edilmiyor, hissedar payı doğrudan "
                     f"eriyor.")
    sh_ = m.get("short") or {}
    spf = sh_.get("float_yuzdesi")
    if spf is not None and spf > 0.20:
        flag("yüksek", f"Yoğun açığa satış: float'ın {fmt_pct(spf)}'i kısa "
                       f"pozisyonda — piyasa güçlü aleyhte bahis oynuyor; "
                       f"aynı anda squeeze (yukarı sıkışma) riski de var. "
                       f"İki ucu keskin.")
    elif spf is not None and spf > 0.10:
        flag("orta", f"Dikkat çekici açığa satış: float'ın {fmt_pct(spf)}'i "
                     f"kısa pozisyonda.")
    return F


# ---------------------------------------------------------------------------
# 7) PAHALILIK SKORU & HÜKÜM  (0 = dip ucuz, 100 = balon)
# ---------------------------------------------------------------------------

def quality_score(m):
    """
    ŞİRKET KALİTESİ (fiyattan TAMAMEN bağımsız): 0 çürük → 100 kusursuz.
    Bileşenler: Piotroski (a3), Altman (a2), ROIC (a2), tahakkuk kalitesi
    (a1), brüt marj trendi (a1), FCF pozitifliği (a1). Bu skor 'iyi şirket
    mi?' sorusuna cevap verir; 'iyi fiyat mı?' sorusuna ASLA karışmaz.
    """
    pts, wts = [], []
    p = m.get("piotroski") or {}
    if p.get("max"):
        pts.append(p["score"] / p["max"] * 100); wts.append(3)
    z = m.get("altman")
    if z is not None:
        # DENETİM BULGUSU C: eski rampa (z−1)/4 MODELDEN BAĞIMSIZDI. Ama Z ve
        # Z'' farklı ÖLÇEKLERDE çalışır (Z'' katsayıları çok daha büyük) ve
        # eşikleri farklıdır: üretim 1.81/2.99, hizmet 1.10/2.60. Sabit rampa
        # aynı ekonomik anlamdaki iki sınıra farklı puan veriyordu — hizmet/
        # teknoloji şirketi kendi tehlike sınırında 2.5 puan alırken üretim
        # şirketi kendi tehlike sınırında 20.2 puan alıyordu (8 kat haksızlık).
        # DÜZELTME: puanlama modelin KENDİ eşiklerine göre normalize edilir →
        # tehlike sınırı = 0, güvenli sınırı = 60, güvenlinin 2 puan üstü = 100.
        _ab = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                               ALTMAN_BOLGE["uretim"])
        pts.append(_ramp(z, [(_ab["tehlike"], 0.0), (_ab["guvenli"], 60.0),
                             (_ab["guvenli"] + 2.0, 100.0)])); wts.append(2)
    r = m.get("roic")
    if r is not None:
        pts.append(min(max(r / 0.20, 0), 1) * 100); wts.append(2)
    ac = m.get("accrual")
    if ac is not None:
        pts.append(100.0 if ac < 0 else max(0.0, 100 - ac * 1000)); wts.append(1)
    gm = m.get("gm") or {}
    if len(gm) >= 3:
        ys = sorted(gm)
        pts.append(100.0 if gm[ys[-1]] >= gm[ys[-3]] - 0.01 else 40.0)
        wts.append(1)
    fb = m.get("fcf_base")
    if fb is not None:
        pts.append(100.0 if fb > 0 else 0.0); wts.append(1)
    if not wts:
        return None
    q = sum(x * w for x, w in zip(pts, wts)) / sum(wts)
    # ── DENETİM BULGUSU D (KRİTİK): kalite ekseni kırmızı bayrakları
    # GÖRMÜYORDU. Sonuç: Beneish RİSKLİ + nakit-kâr makası + alacak
    # bayrağı taşıyan bir şirket kalite 85.8 alıp, fiyatı da ucuzsa
    # verdict_info'dan YEŞİL "KALİTELİ + UCUZ — NADİR BÖLGE" hükmü
    # çıkıyordu. Bayraklar yalnız PAHALILIK skoruna prim ekliyordu —
    # yani muhasebe hilesi şüphesi "şirket kötü" değil "hisse pahalı"
    # diye okunuyordu. Bu, kodun kendi iki-eksen doktrinini bozuyordu.
    # DÜZELTME: bayraklar artık KALİTEYİ de düşürür. Pahalılıktaki prim
    # korunuyor — o ayrı bir şeydir ("bu risk için daha çok iskonto
    # iste"), buradaki ise "bu daha kötü bir işletme" demektir.
    # [KALİBRE EDİLMEMİŞ] yüksek=10, orta=3, tavan 45 puan.
    _ceza = min(45, sum(10 if sev == "yüksek" else 3 if sev == "orta" else 0
                        for sev, _ in (m.get("flags") or [])))
    return float(max(0, min(100, q - _ceza)))


def _ramp(x, pts):
    """Parçalı doğrusal eşleme: pts=[(x0,y0),...] artan x. Uçlarda doyar."""
    if x is None:
        return None
    if x <= pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return float(pts[-1][1])


def expensiveness_score(m):
    """
    DENGELİ kompozit pahalılık skoru (0 ucuz → 100 balon). Tasarım ilkeleri:
      1) TEK bileşen skoru uca SÜRÜKLEYEMEZ — en varsayım-duyarlı bileşen
         olan DCF/MOS'un ağırlığı sınırlıdır.
      1-B) MUTLAK DCF/MOS KARAR DIŞIDIR (horizon uyumsuzluğu) — skor
         ters-DCF + çarpanlar + tarihsel bant üzerinden kurulur.
      2) Çarpanlar MUTLAK bantlarla puanlanır (P/FCF, F/K, EV/EBITDA, PEG).
      3) Tarihsel konum (F/K + F/S bandı, 52h) ve fiyat-değer makası katılır.
      4) Nihai skor [3, 97] aralığına kıstırılır: 0 veya 100 'kesinlik'
         iddia eder — bu model o kesinliğe sahip değildir.
    Claude katmanına 'bu skoru eleştir' talimatı ayrıca verilir.
    """
    comps = []                                # (alt_skor, ağırlık)
    # ── DENETİM DÜZELTMESİ: ağırlık yeniden dengelendi ────────────────
    # ESKİ: MOS/DCF ağırlığı 2.5 ile en yüksekti. Mutlak DCF kaliteli
    # şirketleri sistematik olarak ucuz gösteremediği için bu, hatayı tüm
    # skora enjekte ediyordu (AAPL/MSFT sürekli "pahalı" çıkıyordu).
    # YENİ: MOS ağırlığı 1.0'a indi; yerine TERS-DCF (fiyatın ima ettiği
    # büyüme, şirketin kendi büyümesiyle kıyaslanır) 2.5 ağırlıkla
    # birincil sinyal oldu. Damodaran: Wall Street analizlerinin ~%85'i
    # göreli değerlemeye dayanır; tek-nokta DCF sahte kesinlik üretir.
    # ── QARP GEÇİŞİ: MOS/mutlak DCF bileşeni SKORDAN TAMAMEN ÇIKARILDI.
    # Gerekçe (horizon uyumsuzluğu): mutlak değer sinyali 1-5 yıl ölçeğinde
    # gerçekleşir (LSV 1994: değer primi 5 yılda ~%90), kullanım ufku ise
    # 3 ay — bu pencerede mutlak DCF sinyali gürültüdür. m["mos"] BİLGİ
    # olarak hesaplanmaya devam eder ve UI + payload'da görünür; karar
    # skoruna GİRMEZ. Skor artık ters-DCF + çarpanlar + tarihsel bant.
    # TERS-DCF: fiyat hangi büyümeyi ima ediyor, şirket bunu yapabilir mi?
    _ig, _kendi = m.get("implied_g"), m.get("buyume_kullanilan")
    if _ig is not None and _kendi is not None:
        _makas = _ig - _kendi          # + ise piyasa fazlasını fiyatlıyor
        comps.append((_ramp(_makas, [(-0.10, 12), (-0.04, 30), (0.0, 50),
                                     (0.04, 70), (0.10, 88),
                                     (0.20, 96)]), 2.5))
        m["ters_dcf_makas"] = round(_makas, 4)
    # TARİHSEL F/K BANDI KONUMU — FIX: eski kod `m.get("pe_band")` diye
    # HİÇ VAR OLMAYAN bir anahtarı okuyor, ps_band'de de "pos" değil
    # "konum_0dip_1zirve" bulunuyordu → bu dal ÖLÜYDÜ, 1.5 ağırlıklı
    # bileşen skora hiç girmiyordu. Doğru kaynak forward_bands() çıktısı.
    # 0 = tarihsel dip F/K, 1 = tarihsel zirve F/K.
    _fkb = (((m.get("fwd") or {}).get("fk_tarihsel_bant") or {})
            .get("konum_0dip_1zirve"))
    # ══ SKORDAN ÇIKARILDI → BAĞLAM ═══════════════════════════════════
    # "Bugünkü F/K, şirketin KENDİ tarihsel aralığının neresinde"
    # sorusu tek-hisse çarpan ortalamasına dönüş bahsidir. Kanıt zayıf:
    # F/K ortalamaya döner ama YAVAŞ ve genelde YAPISAL KIRILMALAR
    # etrafında; ortalamanın kendisi kayar. Yapısal olarak yeniden
    # fiyatlanan kaliteli bir şirket sonsuza kadar "kendi geçmişine göre
    # pahalı" görünür → DEĞER TUZAĞI üretici.
    # MSCI, S&P ve FTSE'nin HİÇBİRİ kendi-geçmişi bandı kullanmıyor;
    # hepsi KESİTSEL (akranlara göre) çalışıyor.
    # Bant SİLİNMEDİ — hesaplanmaya ve arayüzde gösterilmeye devam
    # ediyor; sadece SKORU SÜRÜKLEMİYOR.
    if _fkb is not None:
        m["fk_band_konum"] = round(float(_fkb), 3)
    # ══ B9-1 (KRİTİK) ════════════════════════════════════════════════
    # Çarpanlar HER ŞİRKET İÇİN AYNI mutlak eşiklerle puanlanıyordu.
    # Damodaran: "Her çarpanın içinde büyüme, risk ve nakit akışı örüntüleri
    # gömülüdür." ÖLÇÜM (Damodaran Ocak 2026 sektör medyanları): 18x F/K
    # her sektörde 47 puan alıyor ama bölgesel bankada medyanın %64
    # ÜSTÜNDE (pahalı), biyoteknolojide %72 ALTINDA (ucuz). Daha çarpıcısı:
    # sektör MEDYANININ kendisi 25 (hayat sigorta) ile 89 (biyoteknoloji)
    # arası puan alıyor — 64 PUANLIK fark yalnız sektör üyeliğinden ve bu,
    # 35/55/70 hüküm eşiklerinin İKİSİNİ birden aşıyor.
    # ÇÖZÜM: %70 sektör-göreli + %30 mutlak çıpa. Mutlak çıpa, "bütün
    # sektör pahalı" kör noktasını (1999 teknoloji) görünür tutar.
    _sek = m.get("sector")

    def _carpan_puan(deger, anahtar, ramp, w_goreli=0.70):
        _d = _num(deger)
        if _d is None or _d <= 0:
            return None
        _mut = _ramp(_d, ramp)
        _med = (SEKTOR_CARPAN_MEDYAN.get(_sek or "") or {}).get(anahtar) \
            or GENEL_CARPAN_MEDYAN.get(anahtar)
        if not _med or _med <= 0:
            return _mut
        _gor = _ramp(_d / _med, [(0.4, 12), (0.7, 30), (1.0, 50),
                                 (1.5, 68), (2.0, 80), (3.0, 92)])
        return w_goreli * _gor + (1 - w_goreli) * _mut

    # ══ B9-4 SÜTUN GRUPLAMASI (bu tarihe kadar HİÇ UYGULANMAMIŞTI) ═══
    # F/K, EV/EBITDA, P/FCF ve PEG'in DÖRDÜ DE "fiyat ÷ bir kârlılık
    # ölçüsü". Ayrı ayrı ortalamaya girince DÖRT BAĞIMSIZ OY gibi
    # davranıyorlardı — oysa TEK faktörü (kazanç çarpanı) dört kez
    # sayıyorlardı. Eskiden toplam ağırlığın ~%49'unu taşıyorlardı, yani
    # skor süslü bir kılıkta tek faktöre kaldıraçlıydı.
    # OECD/JRC Bileşik Gösterge El Kitabı bunu açıkça ÇİFTE SAYIM diye
    # uyarıyor; çözüm olarak HİYERARŞİK SÜTUN gruplamasını veriyor.
    # MSCI Enhanced Value, S&P Pure Value ve FTSE'nin ÜÇÜ DE böyle
    # yapıyor: 3 metrik, tek değer boyutu, sektör-göreli.
    #
    # SÜTUN İÇİ AĞIRLIK kanıta göre: EV/EBITDA en güçlüsü —
    # Loughran & Wellman (JFQA 2011): kurumsal çarpan faktörü YILLIK
    # %5.28 prim, standart değer oranlarının en yükseği. PEG ise en
    # zayıfı: Damodaran'ın gösterdiği gibi F/K'nın büyümeyle DOĞRUSAL
    # ölçeklendiğini varsayıyor ki bu yanlış; ayrıca E/P ile örtüşüyor.
    _sutun = []          # (puan, sütun_içi_ağırlık)

    # (banka) finansalda CFO−CapEx tabanlı P/FCF anlamsız — sütundan
    # çıkarılır; None ile aşağıdaki ceza dalına da girmez.
    pf = (None if m.get("is_financial")
          else (m.get("p_fcf_adj") if m.get("p_fcf_adj") is not None
                else m.get("p_fcf")))
    if pf is not None and pf > 0:
        _p = _carpan_puan(pf, "p_fcf", [(6, 15), (12, 32), (18, 45),
                                        (28, 62), (45, 80), (70, 93)])
        if _p is not None:
            _sutun.append((_p, 1.2))
    elif (m.get("fcf_base") is not None and m["fcf_base"] <= 0
          and not m.get("is_financial")):
        # ── DÜZELTİLDİ: SABİT CEZA → ANLAMLI ORANA DÜŞ ───────────────
        # Eski hâli negatif FCF'ye düz 78 puan yazıyordu. Sorun: bu,
        # "battı" ile "bilerek yatırım yapıyor"u AYNI kefeye koyuyor.
        # Bugün ABD borsasında şirketlerin %40'tan fazlası negatif kâr
        # bildiriyor ve büyük kısmı Ar-Ge/büyüme tercihi (Amazon örüntüsü),
        # sıkıntı değil. Mohrschladt-Siedhoff (Abacus 2024): zarar eden
        # şirketlerin İÇİNDE bile değer sinyali çalışıyor — yani "ucuz
        # zarar eden" ile "pahalı zarar eden" farklı davranıyor.
        # ÇÖZÜM: payda negatifse ANLAMINI KORUYAN orana geç (EV/Satış),
        # ceza ancak o da yoksa ve SIKINTI İŞARETİ varsa uygulanır.
        _evs = _num(m.get("ev_sales"))
        if _evs is None:
            _ev, _rev = _num(m.get("ev")), latest(m.get("rev") or {})
            if _ev and _rev and _rev > 0:
                _evs = _ev / _rev
        if _evs and _evs > 0:
            m["ev_sales"] = round(float(_evs), 2)   # A-3
            _sutun.append((_ramp(_evs, [(0.8, 20), (2.0, 38), (4.0, 55),
                                        (8.0, 74), (15.0, 90)]), 1.2))
            m["negatif_payda_notu"] = (
                f"FCF negatif → P/FCF hesaplanamıyor. Sabit ceza yerine "
                f"EV/Satış ({_evs:.1f}x) kullanıldı: negatif nakit akışı "
                f"tek başına 'pahalı' demek değildir (şirketlerin %40'tan "
                f"fazlası zarar bildiriyor, çoğu bilerek yatırım yapıyor).")
        else:
            # EV/Satış de yoksa ceza — ama SIKINTI VARSA daha ağır
            _z = _num(m.get("altman"))
            _agir = (_z is not None and _z < 1.81) or bool(
                [f for f in (m.get("flags") or []) if f[0] == "yüksek"])
            _sutun.append((82.0 if _agir else 68.0, 1.2))
            m["negatif_payda_notu"] = (
                "FCF negatif ve EV/Satış da yok → ceza uygulandı"
                + (" (sıkıntı işareti var, ağır)" if _agir
                   else " (sıkıntı işareti yok, hafif)"))

    # ── ARAŞTIRMA TURU A4: TRAILING BİRİNCİL ─────────────────────────
    # yfinance forwardPE bilinen hatalar taşıyor (#1911 sınıfı sürüyor,
    # pegRatio #2570 Haz-2025'ten beri bozuk); Damodaran sektör
    # medyanları TRAILING bazlı (elmayla elma); analist ileri tahminleri
    # sistematik iyimser. Forward atılmadı — ikincil yedek, ve trailing
    # ile >%30 makas açıldığında güven notu düşülür.
    pe = m.get("pe_t") or m.get("pe_f")
    if (m.get("pe_t") and m.get("pe_f") and m["pe_t"] > 0
            and abs(m["pe_f"] - m["pe_t"]) / m["pe_t"] > 0.30):
        m["pe_uyumsuzluk_notu"] = (
            f"Trailing F/K {m['pe_t']:.1f} ile forward F/K "
            f"{m['pe_f']:.1f} arasında >%30 makas — ya hızlı kazanç "
            f"değişimi ya veri hatası (yfinance forwardPE bilinen "
            f"sorunlu). Skor TRAILING ile kuruldu; forward'ı Claude "
            f"katmanında doğrulat.")
    if pe is not None and pe > 0:
        _p = _carpan_puan(pe, "pe", [(7, 18), (12, 33), (18, 47),
                                     (28, 62), (45, 80), (70, 92)])
        if _p is not None:
            _sutun.append((_p, 1.2))
    elif (latest(m.get("ni") or {}) or 0) <= 0:
        # DÜZELTİLDİ (yukarıdaki gerekçenin aynısı): negatif kâr sabit
        # 85 puan yerine önce EV/Satış'a düşer.
        _evs2 = _num(m.get("ev_sales"))
        if _evs2 is None:
            _ev, _rev = _num(m.get("ev")), latest(m.get("rev") or {})
            if _ev and _rev and _rev > 0:
                _evs2 = _ev / _rev
        if _evs2 and _evs2 > 0:
            m["ev_sales"] = round(float(_evs2), 2)   # A-3: teşhis satırı
            _sutun.append((_ramp(_evs2, [(0.8, 22), (2.0, 40), (4.0, 57),
                                         (8.0, 76), (15.0, 90)]), 1.2))
        else:
            _z2 = _num(m.get("altman"))
            _sutun.append((85.0 if (_z2 is not None and _z2 < 1.81)
                           else 72.0, 1.2))

    ee = m.get("ev_ebitda")
    if ee is not None and ee > 0:
        _p = _carpan_puan(ee, "ev_ebitda", [(5, 20), (9, 35), (14, 50),
                                            (22, 65), (32, 80), (45, 92)])
        if _p is not None:
            _sutun.append((_p, 1.6))        # EN GÜÇLÜ oran — öne çıkarıldı
    elif (m.get("ebitda_ttm") is not None and m["ebitda_ttm"] <= 0
          and not m.get("is_financial")):
        _sutun.append((88.0, 1.6))

    # ══ B9-2 (KRİTİK) ════════════════════════════════════════════════
    # Eksik bileşen ATILIP kalanla yeniden ölçekleniyordu — bu yalnız veri
    # TAMAMEN RASTGELE eksikse yansızdır. Buradaki eksiklik ALEYHTE ve
    # rastgele DEĞİL: PEG eksik ⟺ büyüme negatif, EV/EBITDA eksik ⟺
    # EBITDA<0, P/FCF eksik ⟺ FCF<0. Yani bileşenler TAM DA ŞİRKET KÖTÜ
    # OLDUĞU İÇİN düşüyor. ÖLÇÜM: üçü birden eksik şirket eski yolla
    # 69.4 ("PAHALI"), ceza imputasyonuyla 83.5 ("ÇOK PAHALI").
    peg = m.get("peg")
    if peg is not None and peg > 0:
        _sutun.append((_ramp(peg, [(0.6, 25), (1.0, 42), (1.8, 58),
                                   (2.8, 74), (4.0, 88)]), 0.5))
    elif (m.get("earnings_growth_info") is not None
          and m["earnings_growth_info"] <= 0):
        _sutun.append((85.0, 0.5))

    # ── sütunu TEK bileşene indir ────────────────────────────────────
    if _sutun:
        _tw = sum(w for _, w in _sutun)
        _sp = sum(p * w for p, w in _sutun) / _tw
        comps.append((_sp, 3.0))
        m["carpan_sutunu"] = {
            "puan": round(_sp, 1), "adet": len(_sutun),
            "not": ("F/K, EV/EBITDA, P/FCF ve PEG aynı şeyin farklı "
                    "yüzleridir (fiyat ÷ kârlılık). Ayrı ayrı sayılsalardı "
                    "tek faktör dört kez oy kullanırdı; tek sütunda "
                    "birleştirilip TEK ağırlık verildi. Sütun içinde "
                    "EV/EBITDA en ağır — kanıtı en güçlü oran.")}

    # ══ P/S BANDI: SKORDAN ÇIKARILDI → BAĞLAM ════════════════════════
    # "Bugünkü çarpan, şirketin KENDİ tarihsel aralığının neresinde"
    # sorusu tek-hisse çarpan ortalamasına dönüş bahsidir. Kanıt zayıf:
    # çarpanlar ortalamaya döner ama YAVAŞ ve genelde YAPISAL KIRILMALAR
    # etrafında; ortalamanın kendisi kayar. Yapısal olarak yeniden
    # fiyatlanan kaliteli bir şirket sonsuza kadar "kendi geçmişine göre
    # pahalı" görünür → DEĞER TUZAĞI üreticisi. Ayrıca bu bant bugünkü
    # hisse adedini geçmiş yıllara uyguluyor, yani zaten yaklaşık.
    # MSCI, S&P ve FTSE'nin HİÇBİRİ kendi-geçmişi bandı kullanmıyor.
    # SİLİNMEDİ — hesaplanıyor ve arayüzde gösteriliyor, skoru sürüklemiyor.
    band = (m.get("ps_band") or {}).get("konum_0dip_1zirve")
    if band is not None:
        m["ps_band_konum"] = round(float(band), 3)
    pos = m.get("pos_52w")
    # ══ SİLİNDİ — İŞARET TERSTİ ═══════════════════════════════════
    # 52 hafta konumu bir PAHALILIK bileşeni olarak kullanılıyordu:
    # zirveye yakın = daha pahalı (0→38, 1.0→62 puan).
    # George-Hwang (Journal of Finance 2004): zirveye yakınlık DAHA
    # YÜKSEK getiri öngörüyor (+%0.45/ay ham, +%0.85 risk-ayarlı) ve
    # "bu getiriler uzun vadede GERİ DÖNMÜYOR". Yani kod, belgelenmiş
    # bir MOMENTUM etkisini TERS işaretle değerleme skoruna sokuyordu.
    # ÖLÇÜLDÜ: dipte 36.4 → zirvede 37.6 (+1.2 puan yanlış yönde).
    # Küçük etki ama YANLIŞ YÖNDE çalışan bir bileşen, hiç olmamasından
    # kötüdür. Momentum isteniyorsa AYRI eksende ve DOĞRU işaretle
    # gösterilir — değerleme skorunun içinde değil.
    # (m["pos_52w"] hâlâ hesaplanıyor ve arayüzde bağlam olarak duruyor.)
    pv = m.get("pv_gap")
    if pv is not None:
        # SKORDAN ÇIKARILDI → BAĞLAM
        # "Fiyat gelirden N kat hızlı büyümüş" uydurma bir ölçüttür;
        # literatürde karşılığı YOK. Yol-bağımlı, kaba bir çarpan
        # genişlemesi vekili. Bağlam olarak anlamlı, skorda değil.
        m["pv_gap_deger"] = round(float(pv), 2)

    # DENETİM BULGUSU I: eski kod bileşen yoksa `return 62.0` ile ERKEN
    # ÇIKIYORDU — bayrak primi o yola HİÇ uğramıyordu. Sonuç: hiç çarpanı
    # hesaplanamayan (tam da en riskli) bir şirket, 10 tane "yüksek" kırmızı
    # bayrağı olsa bile tam 62.0 alıyordu; bayraksız hâliyle AYNI skor.
    # Bayrakların en çok gerektiği yerde devre dışıydılar.
    # B9-2: kapsama takibi — eksik bileşenler genellikle şirket kötü
    # olduğu için eksiktir; düşük kapsamada tek sayılık hüküm yanıltır.
    _gecerli = [(v, w) for v, w in comps if v is not None]
    m["skor_kapsama"] = (round(len(_gecerli) / max(len(comps), 1), 2)
                         if comps else 0.0)
    if not _gecerli:
        s = 62.0
        m["skor_kapsama_notu"] = (
            "HİÇBİR değerleme bileşeni hesaplanamadı. Bu genellikle kâr, "
            "EBITDA ve FCF'in AYNI ANDA negatif olması demektir — yani "
            "nötr bir durum DEĞİLDİR. Varsayılan 62 bir ÖLÇÜM DEĞİL "
            "TAHMİNDİR; bu skoru düşük güvenli say.")
    else:
        s = sum(v * w for v, w in _gecerli) / sum(w for _, w in _gecerli)
        if m["skor_kapsama"] < 0.50:
            m["skor_kapsama_notu"] = (
                f"Bileşenlerin yalnız %{m['skor_kapsama']*100:.0f}'i "
                f"hesaplanabildi — eksik bileşenler genellikle şirket kötü "
                f"olduğu için eksiktir. Bu skoru DÜŞÜK GÜVENLİ say.")
    sev = sum(4 if f[0] == "yüksek" else 2 for f in m.get("flags", []))
    s += min(sev, 12)
    m["pahalilik_ham"] = round(float(s), 1)   # tavan-yapışması şeffaflığı
    return float(max(3, min(97, s)))


# ══════════════════════════════════════════════════════════════════════
# KALİBRASYON SONUCU — 30 Temmuz 2026, 250 hisse × 16 yıl
# ══════════════════════════════════════════════════════════════════════
# Skor, SEC EDGAR'ın noktasal-zaman (as-filed) verisiyle geçmişe
# hesaplandı; 2.612 gözlem, 2010-2025. Ölçütler veriye BAKILMADAN önce
# ilan edildi. SONUÇ: KALDI (2/4).
#
#   ✖ Monotonluk : Spearman −0.16  (gereken ≤ −0.60)
#   ✖ Makas      : %+5.2, t=1.85   (gereken >0 ve t>2.0)
#   ✔ Örneklem dışı: %+7.3 (içinin %205'i)
#   ✔ Yeterlilik : desil başı 108 hisse · 9 yıl
#
# NE BULUNDU: Sinyal YALNIZCA en ucuz iki desilde. En ucuz %20 → %19.2
# ortalama 12 aylık getiri; kalan %80 → %14.4 (fark +%4.9). Desil 2-9
# arası tamamen düz (%12.6–%15.6 — 3 puanlık gürültü bandı).
#
# SONUÇ: Eşikler (35/55/70/85) KALİBRE EDİLEMEDİ ve KALDIRILDI. Skor bir
# SIRALAMA aracıdır; "şu sayının üstü pahalıdır" iddiası veriyle
# desteklenmiyor. Yalnız en ucuz uçta zayıf bir sinyal var ve o da
# t=1.85 ile anlamlılık eşiğinin altında.
#
# BİLİNEN SINIRLAR: hayatta kalma yanlılığı (sonucu İYİ gösterir),
# skorun ~%55'i test edildi (ters-DCF ve PEG noktasal-zaman kurulamadı),
# 2010-2025 büyük ölçüde tek rejim.
KALIBRASYON_SONUCU = {
    "tarih": "2026-07-30", "hisse": 250, "gozlem": 2612,
    "yil": "2010-2025", "hukum": "KALDI", "gecen_olcut": "2/4",
    "spearman": -0.16, "makas": 0.052, "t_ist": 1.85,
    "ucuz_ceyrek": 0.192, "kalan": 0.144, "fark": 0.049,
    "ozet": (
        "Skor 250 hisse × 16 yıl noktasal-zaman veriyle sınandı ve "
        "ÖNCEDEN İLAN EDİLEN 4 ölçütten 2'sini geçemedi. Desil sıralaması "
        "MONOTON DEĞİL (Spearman −0.16): skor 55 ile 70 arasında ayrım "
        "yapamıyor. Bulunan tek şey: en ucuz %20'lik dilim, kalan %80'e "
        "göre yılda ortalama +%4.9 fazla getirmiş — ama t=1.85, yani "
        "gürültüden ayırt edilemiyor."),
    "sonuc": (
        "EŞİKLER KALDIRILDI. Skor SIRALAMA için kullanılır: 'A, B'den "
        "daha pahalı' demek meşrudur. 'Skor 60, demek ki pahalı' demek "
        "MEŞRU DEĞİLDİR — o iddianın arkasında veri yok."),
}


def verdict_info(score, quality=None):
    """
    İKİ EKSENLİ HÜKÜM MATRİSİ.
    'İyi şirket mi?' (kalite) ile 'iyi fiyat mı?' (skor) AYRI sorulardır.
    Doktrin: Hüküm daima FİYAT ekseninden verilir — kalite, pahalılığın
    mazereti yapılmaz. Mükemmel şirket + pahalı fiyat = alım yeri DEĞİL.
    """
    # ── ETİKETLER BANT TANIMIDIR, HÜKÜM DEĞİL ───────────────────────
    # 30 Tem 2026 kalibrasyonu: eşikler sınandı ve KALDI (2/4 ölçüt).
    # Bu yüzden etiketler artık "PAHALI" gibi bir HÜKÜM değil, "SKOR
    # YÜKSEK BANTTA" gibi bir KONUM bildiriyor. Fark önemli: birincisi
    # doğrulanmamış bir iddia, ikincisi ölçülen bir olgu.
    # DOKTRİN KORUNDU: kalite, yüksek skoru mazur göstermez — bu bir
    # ampirik iddia değil, bilinçli bir disiplin tercihidir.
    hi_q = quality is not None and quality >= 65
    lo_q = quality is not None and quality < 40

    if score < 35:
        if lo_q:
            # ── B9-3 ─────────────────────────────────────────────────
            # Şiddet "warning"den "error"a çıkarıldı. Piotroski (2000):
            # ucuz hisselerin %44'ünden AZI pozitif piyasa-üstü getiri
            # verir; ayıklayan şey KALİTEDİR. UCUZ+ÇÜRÜK en kötü çeyrektir
            # ve eski hâlinde turuncu bir "risk" notuyla geçiliyordu.
            return ("SKOR ALT UÇTA + KALİTE ZAYIF — DEĞER TUZAĞI RİSKİ", "⛔",
                    "error",
                    "EN KÖTÜ ÇEYREK. Piotroski (2000): ucuz hisselerin "
                    "%44'ünden AZI pozitif piyasa-üstü getiri verir; "
                    "ayıklayan şey KALİTEDİR. Düşük kalite + düşük fiyat "
                    "bir fırsat değil, çoğu zaman bir uyarıdır. Ucuzluğun "
                    "NEDENİ cevaplanmadan bu bir alım adayı DEĞİLDİR.")
        if hi_q:
            return ("SKOR ALT UÇTA + KALİTE GÜÇLÜ", "🟢", "success",
                    "Kalite yüksek, fiyat adil değerin altında. Bu nadir "
                    "kombinasyonda bile görev falsifikasyondur: piyasa neyi "
                    "görüyor da iskonto veriyor? En kötü senaryoyu Claude "
                    "katmanında test ettir.")
        return ("SKOR ALT UÇTA — KALİTEYİ DOĞRULA", "🟢", "success",
                "Matematik iskonto gösteriyor. Şimdi soru şu: bu iskonto "
                "fırsat mı, ceza mı? Ucuzluk tuzağı ihtimalini sorgulat.")
    if score < 55:
        return ("SKOR ORTA BANTTA", "🟡", "info",
                "Fiyat, adil değer bandının içinde. Kenar (edge) değerlemeden "
                "değil, işin kalitesinden ve katalizörlerden gelmek zorunda."
                + (" Kalite tarafı güçlü — izleme listesi adayı."
                   if hi_q else ""))
    if score < 70:
        if hi_q:
            # ── B9-3 ─────────────────────────────────────────────────
            # Asness-Frazzini-Pedersen "Quality Minus Junk" (2019): kalite
            # FİYATLANMIŞ bir özelliktir ve piyasa onu EKSİK fiyatlar —
            # kaliteli-eksi-çürük portföy aylık 71-97 baz puan anormal
            # getiri üretir. Novy-Marx: kalite+değer birlikte tek başına
            # her ikisinin İKİ KATI. Bu yüzden ILIMLI prim otomatik RET
            # değildir — ama ispat yükü kalitenin KALICILIĞINDADIR.
            # DOKTRİN KORUNDU: balon fiyat aşağıda hâlâ DUR diyor.
            return ("SKOR YÜKSEK + KALİTE GÜÇLÜ — İSPAT YÜKÜ KALİTEDE", "🟡",
                    "warning",
                    "İki eksen ayrıştı ve bu OTOMATİK BİR RET DEĞİLDİR: "
                    "kalite fiyatlanmış bir özelliktir ve piyasa onu eksik "
                    "fiyatlar (QMJ: aylık 71-97bp anormal getiri). Ilımlı "
                    "bir prim savunulabilir — AMA ispat yükü sendedir: "
                    "kalitenin KALICI olduğunu (ROIC serisi, marj istikrarı, "
                    "hendek kanıtı) gösteremiyorsan bu yalnızca pahalı bir "
                    "hissedir. Kademeli giriş burada bir SABIR aracıdır, "
                    "alım çağrısı değil.")
        return ("SKOR YÜKSEK BANTTA", "🟠", "warning",
                "Fiyat adil değerin üzerinde. Bu seviyeden alım, güvenlik "
                "marjı olmadan risk almaktır.")
    if score < 85:
        if hi_q:
            return ("SKOR ÇOK YÜKSEK + KALİTE MÜKEMMEL — YİNE DE DUR", "🔴",
                    "error",
                    "İki eksen ayrıştı: işletme kusursuza yakın, fiyat ise "
                    "matematiksel adil değerin çok üzerinde. İyi şirket ≠ iyi "
                    "hisse. Bu makasta risk-ödül alıcının aleyhinedir; "
                    "yatırımcı için doğru hamle beklemektir.")
        if lo_q:
            return ("SKOR ÇOK YÜKSEK + KALİTE ZAYIF — DUR", "⛔", "error",
                    "Kalite düşük, fiyat şişkin: yatırımcı için en kötü "
                    "kombinasyon. Burada tartışılacak bir tez yok.")
        return ("SKOR ÇOK YÜKSEK BANTTA — DUR", "🔴", "error",
                "DİSİPLİN KURALI DEVREDE: Fiyat matematiksel adil değerin "
                "belirgin üzerinde. Kalite, pahalılığın mazereti değildir. "
                "Bu seviyede risk-ödül yatırımcının aleyhine dönmüştür.")
    if lo_q:
        return ("SKOR UÇ BANTTA + KALİTE ZAYIF — DUR", "⛔", "error",
                "Zayıf işletme, spekülatif fiyat. Momentum döndüğünde ilk ve "
                "en sert düşen bu profildir.")
    label = "SKOR UÇ BANTTA — DUR"
    msg = ("Fiyat, savunulabilir hiçbir nakit akışı senaryosuyla "
           "gerekçelendirilemiyor. Bu bölgede fiyatı temel analiz değil, "
           "hikâye ve momentum taşır — ve momentum döndüğünde ilk düşen, "
           "güvenlik marjı olmayandır.")
    if hi_q:
        label += " (KALİTE İYİ, SKOR YİNE DE UÇTA)"
        msg = ("Şirket operasyonel olarak kusursuza yakın — ama bu, hisseyi "
               "kurtarmaz: " + msg)
    return (label, "⛔", "error", msg)


# ---------------------------------------------------------------------------
# 8) CLAUDE KATMANI — payload + API çağrısı
# ---------------------------------------------------------------------------

def _musd(d):
    """{yıl→değer} → {yıl→milyon, 1 ondalık} (token tasarrufu)."""
    return {str(y): (round(v / 1e6, 1) if v is not None else None)
            for y, v in sorted((d or {}).items())}


# ── B10-3: PROMPT ↔ PAKET SÖZLEŞMESİ ─────────────────────────────────
# Talimat, paketteki alanları ADIYLA istiyor. Alan yeniden adlandırılır ya
# da kaldırılırsa model bunu BİLDİRMEZ, UYDURUR — bu, bu turun en yüksek
# riskli yapısal açığıdır (FaithEval: bağlam yokken doğru "bilmiyorum"
# oranı bir modelde %76.8 → %7.4). Elle isim düzeltmek geçici çözümdür;
# kalıcı çözüm, GERÇEKTEN VAR OLAN alan listesini her çağrıda modele
# söylemektir.
TALIMATIN_BEKLEDIGI_ALANLAR = [
    "celiski_detektoru", "kural_motoru_kirmizi_bayraklar",
    "veri_tutarlilik_denetimi", "sec_edgar_dogrulama",
    "monte_carlo_adil_deger", "senaryo_beklenen_deger", "epv_taban_deger",
    "dupont_roe_ayristirma", "nakit_donusum_dongusu", "tuzak_uyarilari",
    "sektor_sablonu", "sermaye_tahsisi", "baz_oranlar", "kazanc_cagrisi",
    "normalize_kazanc", "ileri_tahmin_ve_bantlar", "kademeli_plan",
    "cikis_plani", "tutma_ufku", "plan_saglamasi", "stop_ve_destek",
    "seviye_senaryolari", "rakip_kiyasi", "akis_ve_katalizor",
    "haber_ve_olaylar", "rejim_ve_zamanlama", "onceki_analiz_kayitlari",
    "bakim_buyume_capex_ayrimi", "beneish_m_skoru", "veri_kalitesi",
]


def payload_sozlesmesi(p):
    """Modele HANGİ alanların dolu, hangilerinin BOŞ olduğunu söyle.

    Amaç uydurmayı zorlaştırmak: "bu alan pakette YOK" bilgisi açıkça
    verilirse model "veri yok" demeyi seçmesi kolaylaşır. Bu, talimata
    "bilmiyorsan BİLİNMİYOR yaz" demenin YAPISAL karşılığıdır — sadece
    talimatla söylemek yetmiyor (araştırma: açıkça talimat verilse bile
    çekimser kalmak zor).
    """
    dolu, bos = [], []
    for k in TALIMATIN_BEKLEDIGI_ALANLAR:
        v = p.get(k)
        (bos if (v is None or v == {} or v == [] or v == "") else dolu
         ).append(k)
    return ("<paket_sozlesmesi>\n"
            "Bu koşuda paketin DOLU alanları: "
            + (", ".join(dolu) if dolu else "(hiçbiri)") + "\n"
            "Bu koşuda BOŞ/EKSİK alanlar: "
            + (", ".join(bos) if bos else "(yok)") + "\n"
            "KURAL: BOŞ listedeki bir alana dayanan analiz İSTENİYORSA, o "
            "bölümde 'veri yok' yaz ve GEÇ. O alan için sayı, tahmin ya da "
            "'muhtemelen' üretme — eksik alanı doldurmak bu raporun en "
            "tehlikeli hata biçimidir.\n"
            "</paket_sozlesmesi>")


def build_payload(m):
    a = m["assumptions"]
    p = {
        "meta": {
            "ticker": m["ticker"], "sirket": m["name"], "sektor": m["sector"],
            "endustri": m["industry"], "fiyat_ccy": m["ccy"],
            "is_ozeti": (m.get("is_ozeti") or "")[:1500] or None,
            "tablo_ccy": m["fin_ccy"], "ccy_uyumsuz": m["ccy_mismatch"],
            "veri_kaynagi": "Yahoo Finance / yfinance (SEC dipnot derinliği DEĞİL)",
            "cekilme_zamani": m["fetched_at"],
            "egitici_mod": bool(m.get("edu")),
        },
        "fiyat_ve_degerleme": {
            "fiyat": m["price"], "piyasa_degeri": m["mcap"], "ev": m["ev"],
            "hisse_sayisi": m["shares"],
            "dcf_adil_deger": m["fair"], "guvenlik_marji": m["mos"],
            # CANLI BULGU: absürt DCF (ör. fiyat 91.67 iken adil değer 0.60)
            # motor tarafından doğru şekilde YOK SAYILIYOR ama pakete girip
            # Claude'un önüne gidiyordu. Artık açıkça damgalanıyor.
            "dcf_kullanilabilir_mi": bool(
                m.get("fair") and m.get("price")
                and _fair_makul_mu(m["fair"], m["price"])),
            "dcf_uyarisi": (
                None if (m.get("fair") and m.get("price")
                         and _fair_makul_mu(m["fair"], m["price"]))
                else "ABSÜRT/EKSİK — bu DCF sayısını ve güvenlik marjını "
                     "RAPORDA KULLANMA; makul bant fiyatın %25-%300'üdür. "
                     "Değerleme için ters-DCF ve çarpanları kullan."),
            "dcf_adil_deger_araligi": {
                "kotumser": m.get("fair_lo"), "baz": m["fair"],
                "iyimser": m.get("fair_hi"),
                "not": "Adil değer tek sayı değil ARALIKTIR; raporda daima "
                       "aralık olarak an.",
            },
            "ters_dcf_ima_edilen_yillik_fcf_buyumesi": (
                None if m["implied_capped"] else m["implied_g"]),
            "ters_dcf_sinira_carpti": m["implied_capped"],
            "ters_dcf_makas": m.get("ters_dcf_makas"),   # skorun EN AĞIR bileşeni (2.5) — görünür olmalı
            "ters_dcf_yorum": (
                (("Fiyat o kadar yüksek ki modelin üst sınırı (yılda %60 "
                  "büyüme) bile yetmiyor — hiçbir gerçekçi büyüme senaryosu "
                  "bu fiyatı taşımıyor. Sayı verme; bu çerçeveyi kullan.")
                 if (m["implied_g"] or 0) > 0 else
                 ("Fiyat o kadar düşük ki model alt sınıra dayandı — piyasa "
                  "fiilen kalıcı küçülme fiyatlıyor. Ya derin ucuzluk ya da "
                  "piyasanın bildiği bir şey var; falsifikasyon şart."))
                if m["implied_capped"] else None),
            "python_skoru_0ucuz_100balon": m["score"],
            # Denetim Bulgu 1b/5: aşağıdaki alanlar üretiliyor ama hiçbir
            # kanala çıkmıyordu — Claude, skoru güven bağlamı olmadan ve
            # adil değeri sessiz kesintiyle alıyordu. Artık pakette.
            "skor_kapsama_orani": m.get("skor_kapsama"),
            "skor_kapsama_notu": m.get("skor_kapsama_notu"),
            "skor_carpan_sutunu": m.get("carpan_sutunu"),
            "skor_negatif_payda_notu": m.get("negatif_payda_notu"),
            "fk_secim_uyumsuzluk_notu": m.get("pe_uyumsuzluk_notu"),
            "iflas_distres_kesintisi": m.get("distres"),
            "iflas_kesintisi_kapsami": (
                None if not m.get("distres") else
                "dcf_adil_deger + aralık + monte_carlo + epv + duyarlılık "
                "matrisi bu kesintiyi İÇERİR; ters-DCF içermez (fiyattan "
                "büyüme çıkarımıdır, değer tahmini değil)"),
            "terminal_reinvestman_notu": m.get("terminal_reinvest_notu"),
            "kalite_skoru_0curuk_100saglam": m.get("quality"),
            "python_hukmu": m["verdict"][0],
            "dcf_varsayimlari": {
                "buyume": a["growth"], "iskonto": a["discount"],
                "terminal": a["terminal"], "yil": a["years"],
                "sbc_dusuldu": a["sbc_adjust"],
                "net_borc_dusuldu": a["net_debt_adjust"],
                "fcf_baz_kaynak": m["fcf_source"],
                "akis_kimligi": m.get("dcf_akis_kimligi"),
                "banka_uyarisi": m.get("banka_dcf_uyarisi"),
                "roic_terminal": m.get("roic_terminal_kullanilan"),
            },
            "carpanlar": {
                "pe_trailing": m["pe_t"], "pe_forward": m["pe_f"],
                "ps": m["ps"], "pb": m["pb"], "ev_ebitda": m["ev_ebitda"],
                "p_fcf_ham": m["p_fcf"], "p_fcf_sbc_duzeltilmis": m["p_fcf_adj"],
                "peg": m["peg"], "peg_kaynak": m["peg_src"],
                "pe_trailing_kaynak": m.get("pe_t_kaynak"),
            },
        },
        "celiski_detektoru": {
            "aciklama": "Söylenen ile yapılan, fiyat ile temel arasındaki "
                        "makaslar. Anomali buradaysa derinleşme ekseni "
                        "burasıdır.",
            "fiyat_2y_degisim": m.get("price_chg_2y"),
            "gelir_2y_degisim": m.get("rev_chg_2y"),
            # (temizlik) çift anahtar kaldırıldı: "fiyat_deger_makasi_kat"
            # aşağıda pv_gap_deger ile yeniden tanımlanıp bunu eziyordu.
            "fiyat_52h_konum_0dip_1zirve": m.get("pos_52w"),
            "fs_tarihsel_band_yaklasik": m.get("ps_band"),
            "fk_tarihsel_band_konum_0dip_1zirve": m.get("fk_band_konum"),
            "fs_tarihsel_band_konum_0dip_1zirve": m.get("ps_band_konum"),
            "fiyat_deger_makasi_kat": m.get("pv_gap_deger"),
            "arge_gelir_seri": {str(y): round(v, 4) for y, v in
                                sorted((m.get("rnd_rev") or {}).items())},
            "sbc_gelir_seri": {str(y): round(v, 4) for y, v in
                               sorted((m.get("sbc_rev_seri") or {}).items())},
        },
        "buyume_ve_marjlar": {
            "gelir_cagr_3y": m["rev_cagr3"], "fcf_cagr_3y": m["fcf_cagr3"],
            "yahoo_kar_buyumesi_yoy": m["earnings_growth_info"],
            "brut_marj": {str(y): round(v, 4) for y, v in sorted(m["gm"].items())},
            "faaliyet_marji": {str(y): round(v, 4) for y, v in sorted(m["om"].items())},
            "net_marj": {str(y): round(v, 4) for y, v in sorted(m["nm"].items())},
            "roe": m["roe"], "roic": m["roic"],
            "roic_ham_duzeltmesiz": m.get("roic_ham"),
            "roic_kaynak": m.get("roic_kaynak"),
            "roic_arge_kapitalizasyonu": m.get("roic_arge"),
            "roic_detay": m.get("roic_detay"),
            "vergi_kaynak": m.get("vergi_kaynak"),
            "efektif_vergi_defter": m.get("eff_tax_defter"),
            "geri_alim_kalitesi": m.get("geri_alim"),
            "roe_hesap_bazi": m.get("roe_basis"),
            "gpoa_novy_marx": m.get("gpoa"),
            "gpoa_yorum": m.get("gpoa_yorum"),
        },
        "nakit_akisi_musd": {
            "cfo": _musd(m["cfo"]), "capex": _musd(m["capex"]),
            "fcf": _musd(m["fcf"]), "sbc": _musd(m["sbc"]),
            "ttm_fcf_ham": round(m["ttm_fcf_raw"] / 1e6, 1) if m["ttm_fcf_raw"] else None,
            "ttm_ciro": round(m["ttm_rev"] / 1e6, 1) if m.get("ttm_rev") else None,
            "ttm_net_kar": round(m["ttm_ni"] / 1e6, 1) if m.get("ttm_ni") else None,
            "ttm_sbc": round(m["ttm_sbc"] / 1e6, 1) if m["ttm_sbc"] else None,
            "dcf_fcf_bazi": round(m["fcf_base"] / 1e6, 1) if m["fcf_base"] else None,
        },
        "gelir_tablosu_musd": {
            "gelir": _musd(m["rev"]), "brut_kar": _musd(m["gp"]),
            "faaliyet_kari": _musd(m["oi"]), "net_kar": _musd(m["ni"]),
        },
        "bilanco": {
            "toplam_borc": m["total_debt"], "nakit": m["cash"],
            "net_borc": m["net_debt"], "ozsermaye": m["equity"],
            "cari_oran": m["current_ratio"],
            "net_borc_ebitda": m["nd_ebitda"], "faiz_karsilama": m["int_cov"],
        },
        "kalite_ve_adli": {
            "piotroski_f": m["piotroski"], "altman_z": m["altman"],
            # KANIT TURU: model adı EKSİKTİ — Claude Z'yi klasik eşiklerle
            # (1.81/2.99) yorumluyordu; Z'' eşikleri 1.10/2.60'tır.
            "altman_model": m.get("altman_model"),
            "altman_esikleri": ALTMAN_BOLGE.get(
                m.get("altman_model") or "uretim", ALTMAN_BOLGE["uretim"]),
            "altman_uyarisi": m.get("altman_x4_uyarisi"),
            "piotroski_evren_notu": m.get("piotroski_evren_notu"),
            "piotroski_kullanim_modu": m.get("piotroski_mod"),
            "piotroski_kullanim_notu": m.get("piotroski_notu"),
            "altman_finansal_sektorde_atlandi": m["is_financial"],
            "tahakkuk_orani": m["accrual"],
            "dso_gun": {str(y): round(v, 1) for y, v in sorted((m["dso"] or {}).items())},
            "hisse_dilusyonu": {
                "yillik_temiz_medyan": m["dilution_yoy"],
                "yillik_seri": {str(y): round(v, 4) for y, v in
                                sorted((m.get("dilution_seri") or {}).items())},
                "ipo_donusum_sicramasi": m.get("dilution_ipo_jump"),
                "olcum": "yıllık ORTALAMA seyreltilmiş hisse değişimi "
                         "(dönem sonu hisse DEĞİL; IPO sıçraması ayıklandı)",
            },
            "sbc_gelir": m["sbc_rev"],
        },
        "beneish_m_skoru": m.get("beneish"),
        "veri_kalitesi": {
            "ttm_kaynak": m.get("ttm_kaynak"),
            "temel_kaynak": m.get("temel_kaynak"),
            "sec_mutabakati": m.get("mutabakat"),
            "yabanci_dosyalayici_adr": m.get("adr_notu"),
            "sbc_yontemi": m.get("sbc_yontem"),
            "ev_kopru_notu": m.get("ev_kopru_notu"),
        },
        "haber_ve_olaylar": {
            "ima_edilen_hareket": m.get("implied_move"),
            "sec_8k_olaylar": m.get("olaylar_8k"),
            "haber_akisi": m.get("haberler"),
        },
        "akis_ve_katalizor": {
            "katalizor_takvimi": m.get("catalysts"),
            "pead_sinyali": m.get("pead"),
            "analist_revizyon_momentumu": m.get("revizyon"),
            "iceriden_islemler": m.get("insider"),
            "aciga_satis": m.get("short"),
            "kurumsal_sahiplik": m.get("kurumsal"),
            "ipo_lockup": m.get("ipo"),
        },
        "rejim_ve_zamanlama": {
            "momentum_12_1": m.get("momentum"),
            "pead": m.get("pead"),
            "ma200_ustu": m.get("trend_ok"),
            "rejim_uyari": m.get("rejim_uyari"),
            "not": ("Skora girmez; kademeli girişin TEMPOSUNU yönetir "
                    "(Jegadeesh-Titman 1993; Bernard-Thomas 1989).")},
        "monte_carlo_adil_deger": m.get("mc"),
        "senaryo_beklenen_deger": m.get("scenario"),
        "epv_taban_deger": m.get("epv"),
        "bakim_buyume_capex_ayrimi": m.get("capex_ayrim"),
        "dupont_roe_ayristirma": m.get("dupont"),
        "nakit_donusum_dongusu": m.get("ccc"),
        "tuzak_uyarilari": m.get("trap"),
        "sektor_sablonu": m.get("sector_tpl"),
        "sermaye_tahsisi": m.get("capalloc"),
        "baz_oranlar": m.get("baserate"),
        "kazanc_cagrisi": m.get("call"),
        "sec_dosyalamalari": m.get("filings"),
        "normalize_kazanc": m.get("norm"),
        "ileri_tahmin_ve_bantlar": m.get("fwd"),
        "onceki_analiz_kayitlari": m.get("gecmis"),
        "kademeli_plan": m.get("kademe"),
        "cikis_plani": m.get("cikis"),
        "iki_kar_alma_sistemi_uyarisi": (
            (m.get("cikis") or {}).get("iki_sistem_notu")),
        "tutma_ufku": m.get("ufuk"),
        "plan_saglamasi": m.get("plan_saglama"),
        "plan_saglamasi_notu": (
            "Motorun KENDİ ürettiği seviyeleri denetlemesidir (merdiven "
            "sırası, stop/hedef matematiği, bölge tutarlılığı, ulaşılabilirlik). "
            "DOLU ise seviyeler tutarsızdır: raporda ALIM SEVİYESİ VERME, "
            "sorunu öne al ve kullanıcıyı uyar."),
        "stop_ve_destek": {"destek_direnc": m.get("sr"),
                           "stop_plani": m.get("stop"),
                           "veri_tazeligi": m.get("tazelik")},
        "seviye_senaryolari": m.get("seviye_senaryo"),
        "rakip_kiyasi": m.get("peers"),
        "sec_edgar_dogrulama": m.get("edgar"),
        "veri_tutarlilik_denetimi": [
            {"seviye": s, "not": t} for s, t in (m.get("sanity") or [])
        ],
        "kural_motoru_kirmizi_bayraklar": [
            {"siddet": s, "bulgu": t} for s, t in m["flags"]
        ],
    }
    return _clean(p)


CLAUDE_SYSTEM = """Sen, Wall Street'te milyar dolarlık portföyler yönetmiş, \
değerleme ve adli finans (forensic accounting) alanında otorite, kıdemli bir \
analiz partnerisin. Üslubun: keskin, net, veriye dayalı, pazarlama cilasından \
arınmış. Doktrinin tek cümledir: HARİKA ŞİRKETİ YANLIŞ FİYATTAN ALMAK, EN \
PAHALI HATADIR. Şirket dünyanın en dürüst nakit makinesi olsa bile, fiyat \
adil değerin üzerindeyse o hisse alım için uygun yerde DEĞİLDİR ve bunu \
açıkça söylersin.

Sana bir Python motorunun ürettiği yapılandırılmış veri paketi (JSON) \
verilecek. Mutlak kuralların:

1) FİYAT-ÖNCE HÜKÜM (zorunlu açılış): Rapora tek cümlelik, kaçamaksız \
hükümle başla. Kalite skoru yüksek ama pahalılık skoru da yüksekse açılış \
kalıbın şudur: "ŞİRKET MÜKEMMEL AMA FİYAT PAHALI — ALIM İÇİN UYGUN YERDE \
DEĞİL." Kalite, pahalılığın mazereti YAPILMAZ; hüküm daima fiyat ekseninden \
verilir. UCUZ / MAKUL / PAHALI + alım yeri uygun mu + risk-ödül kimin lehine.

2) KAYNAK HİYERARŞİSİ: Resmî tablo rakamı (10-K/10-Q kaynaklı) = GERÇEK. \
Yönetimin 'Düzeltilmiş FAVÖK'ü, yatırımcı sunumu projeksiyonları = İDDİA. \
İddiaları gerçek gibi kullanma; kullanırsan İDDİA diye etiketle. Web arama \
aracın varsa kritik rakamları en güncel resmî dosyalamalardan (SEC) çapraz \
doğrula ve bulduğuna TARİH ver; aracın yoksa bunu açıkça söyle.

3) ÇELİŞKİ DETEKTÖRÜ: Paketteki 'celiski_detektoru' bölümü hammaddendir. \
Söylenen-yapılan çelişkisi (ör. büyüme hikâyesi vs Ar-Ge/Gelir düşüşü + \
SBC/Gelir artışı) ve fiyat-değer çelişkisi (büyüme %X iken fiyat %Y koşmuş; \
çarpan tarihsel bandın neresinde) — bulduğun çelişkiyi ORANIYLA yaz: \
"fiyat, geliri ~N kat aşmış; makas rasyonel değil" gibi. Derinleşme eksenini \
şablondan değil, en büyük anomaliden KENDİN seç ve tek cümleyle gerekçele.

4) TERS-DCF SORGUSU: Fiyata gömülü büyümeyi tarihsel gelir/FCF temposuyla \
kıyasla. Piyasa gerçekçi bir gelecek mi, masal mı fiyatlıyor — sayıyla göster.

5) FALSİFİKASYON (zorunlu bölüm): Ana tezini TERSİNDEN test et. Ucuz/alım \
yeri diyorsan "bu neden düşer/batar?" sorusuna, pahalı diyorsan "beni ne \
haksız çıkarır?" sorusuna en güçlü karşı-kanıtla cevap ver. En kötü senaryoyu \
(kalıcı sermaye kaybı ihtimali) açıkça yaz.

6) MODELİ DENETLE: Python'un DCF varsayımlarını ve sezgisel skorlarını \
sorgula; nerede yanılıyor olabilir, söyle. Sen skorun kölesi değil, \
denetçisisin.

7) ÜÇLÜ DOĞRULUK ETİKETİ (her önemli iddiada): [KESİN] resmî tablodan gelen \
rakam ve düz matematik · [TÜRETİLMİŞ] rakamların işaret ettiği sinyal/çıkarım \
· [HİPOTEZ] doğrulanmamış anlatı, piyasa efsanesi, köpük. Bilmediğini \
uydurma; emin değilsen HİPOTEZ de ve nasıl doğrulanacağını söyle.

8) EĞİTİCİ MOD: meta.egitici_mod true ise her teknik terimi (DCF, PEG, \
tahakkuk, ROIC...) ilk geçtiği yerde tek cümlelik sade Türkçeyle açıkla; \
false ise açıklama ekleme, profesyonel yoğunlukta yaz.

9) SAYI DİSİPLİNİ: (a) Adil değeri daima ARALIK olarak an \
(dcf_adil_deger_araligi: kötümser–iyimser); tek sayıya kesinlik yükleme. \
(b) 'fiyat_ve_degerleme.ters_dcf_sinira_carpti' true ise SAYI AKTARMA; 'ters_dcf_yorum' cümlesini \
kullan. (c) Dilüsyonu YALNIZCA 'hisse_dilusyonu' bölümünden yorumla; \
IPO/dönüşüm sıçramasını sürekli dilüsyon sayma. (d) ROIC'i 'roic' alanından \
al (NOPAT/yatırılan sermaye); ROE ile karıştırma, ikisini aynı cümlede net \
ayır. (e) 'veri_tutarlilik_denetimi' notlarını Veri Sınırları bölümünde ele \
al; uyuşmazlık taşıyan her metriği [HİPOTEZ]'e düşür. (f) \
'sec_edgar_dogrulama': uyum '✓' olan kalemler resmî 10-K TEYİTLİDİR → \
gönül rahatlığıyla [KESİN] kullan; '✗' kalemlerde SEC değeri ESASTIR — \
Yahoo rakamını [HİPOTEZ]'e düşür ve metinde SEC rakamını kullan.

10) KATALİZÖR VE AKIŞ: 'katalizor_takvimi' yalnızca kazanç/temettü içerir; \
şirkete özgü olayları (sözleşme yenilemesi, regülasyon kararı, ürün lansmanı, \
dava, lock-up) web aramasıyla KENDİN bul ve TARİHİYLE yaz — önümüzdeki 90 \
günün olay haritasını çıkar. 'iceriden_islemler'de 10b5-1 (planlı satış) \
ayrımı yoktur; akış tek yönlüyse Form 4 dipnotlarını webden kontrol et. \
'analist_revizyon_momentumu' yönünü kendi tezinle çarpıştır: sen ucuz derken \
revizyonlar aşağıysa bu gerilimi açıkça tart. Beneish 'bolge' RİSKLİ ise \
adli derinleşme eksenini oraya kur ve hangi bileşenin (DSRI/TATA/SGI...) \
skoru sürüklediğini söyle. 'aciga_satis' %10+ ise İKİ YÖNLÜ yorumla (aleyhte \
bahis + squeeze). 'kurumsal_sahiplik' ~45 gün gecikmelidir; trend olarak \
oku. 'ipo_lockup' TAHMİNÎDİR — genç IPO'da gerçek lock-up şartını 424B4 \
prospektüsünden weble doğrula. Web açıkken son KAZANÇ ÇAĞRISININ dilini de \
tara: rehber ifadeleri, 'headwinds/challenging' türü hedge kelimeleri, ton \
değişimi — transcript bulamazsan bulamadığını açıkça söyle.

11) KADEMELİ GİRİŞ / BEKLEME PLANI (zorunlu bölüm): Paketteki \
'kademeli_plan' iskeletini kendi analizinle RAFİNE et — dilim seviyelerini \
katalizör takvimi ve falsifikasyon eşiklerinle gerekçele; pahalıysa plan \
verme, izleme eşiği ver. 'monte_carlo_adil_deger' dağılımı genişse bunu \
değerleme belirsizliği olarak açıkça yaz. 'rakip_kiyasi' doluysa tabloyu \
kullan; boşsa en yakın 3-5 rakibi webden KENDİN bul ve çarpan kıyası yap. \
Plan TAVSİYE DEĞİLDİR, disiplin çerçevesidir — bunu belirt. İzleme eşiği ve \
dilim seviyeleri FİYAT-ÇIPALIDIR; DCF adil değeri yalnızca makul banttaysa \
(fiyatın %25–%300'ü) çapraz-kontrol olarak katılmıştır. Eşiği yorumlarken \
mevcut fiyata göre MANTIKLI olduğunu doğrula; 'esik_kaynagi'/'cipa_notu' \
alanlarını oku ve eşiğin neden o seviyede olduğunu tek cümleyle açıkla. \
Absürt tek-DCF sayısına ASLA geri dönme.

12) TEZ SENTEZİ, KALİTE SÜRÜCÜLERİ VE SENARYO (Aşama 1 — zorunlu): \
(a) YATIRIM TEZİ: Raporun başında (hükümden hemen sonra) tek cümlelik, TEST \
EDİLEBİLİR bir tez kur ve VARIANT PERCEPTION'ı açıkça yaz — 'piyasa neyi \
yanlış fiyatlıyor, ben neye inanıyorum, konsensüsten farkım ne?'. Sonra 3-5 \
İZLENEBİLİR ÖNCÜL listele (tez doğruysa şunlar gerçekleşmeli) ve her birinin \
nasıl izleneceğini söyle. (b) PRE-MORTEM: Falsifikasyon bölümünde 'bu yatırım \
3 yıl sonra battı — en olası 3 neden' analizi yap; mümkünse benzer \
şirketlerin/sektörün tarihsel BASE RATE'ini (taban oran) referans al. \
(c) DuPont: 'dupont_roe_ayristirma'yı kullanıp ROE'nin NEDENİNİ yaz — marj mı, \
varlık devri mi, kaldıraç mı sürüyor; yüksek kaldıraçla şişmiş ROE'yi kırılgan \
olarak işaretle. (d) EPV: 'epv_taban_deger'i büyümesiz TABAN olarak yorumla — \
fiyat EPV'nin çok üstündeyse 'tüm değer büyüme beklentisinde, kırılgan' de; \
EPV DCF ile birlikte üçgenleme sağlar. (e) SENARYO: 'senaryo_beklenen_deger' \
olasılık-ağırlıklı beklenen değeri fiyatla karşılaştır; beklenen değer \
belirginse yorumla, olasılıkları kendi analizinle revize edebilirsin. \
(f) TUZAKLAR: 'tuzak_uyarilari' (çevrimsel zirve tuzağı, değer tuzağı) ve \
'nakit_donusum_dongusu' bulgularını ilgili bölümlerde işle; çevrimsel bir \
şirkette tek-yıl kazancına dayanma, normalize (mid-cycle) bakış öner.

13) SEKTÖR MERCEĞİ, SERMAYE TAHSİSİ, MOAT VE DIŞ GÖRÜŞ (Aşama 2 — zorunlu): \
(a) 'sektor_sablonu' varsa o sektörün DOĞRU merceğini kullan (SaaS→Rule of \
40/NRR; banka→ROTCE, P/TBV, kimlik P/TBV≈F/K×ROTCE; çevrimsel→NORMALİZE F/K \
esastır, raporlanan F/K'yı zirve yanılsamasına karşı sorgula) ve şablondaki \
'claude_gorevleri'ni webden YAP. (b) 'sermaye_tahsisi'ni yorumla: Buffett \
testi, dağıtımın FCF karşılaması, ROIC trendi — CEO parayı değer yaratan \
yere mi koyuyor? Buna bağlı MOAT SINIFLANDIR: Geniş/Dar/Yok + kaynak (ağ \
etkisi, geçiş maliyeti, maliyet avantajı, gayri maddi varlık, verimli ölçek) \
+ kanıt ('roic_seri' kalıcılığı, brüt marj istikrarı); moat iddiasını ROIC \
kanıtıyla çaprazla. (c) 'baz_oranlar' DIŞ GÖRÜŞtür: içeriden bakışla \
çarpıştır; hedef büyüme tarihsel olarak nadirse bunu tezin ispat yüküne \
ekle. (d) SEGMENT & SOTP: 'sec_dosyalamalari' linklerinden son 10-K'nın \
segment dipnotunu webden çıkar — segment gelir/marj tablosu kur, hangi \
segment değeri taşıyor söyle; çok segmentliyse kaba parçaların-toplamı \
(SOTP) dene. (e) 'kazanc_cagrisi' doluysa dil analizini derinleştir \
(guidance yönü, hedge yoğunluğu, ton değişimi) ve söylem-gerçek \
dedektörüyle birleştir; boşsa son çağrının dilini webden ara.

14) NORMALİZE KAZANÇ, İLERİYE BAKIŞ VE KALİBRASYON (Aşama 3 — zorunlu): \
(a) 'normalize_kazanc': 'durum' ZİRVE ÜSTÜ ise raporlanan kârla yapılan TÜM \
çarpanları normalize F/K ile yeniden test et ve zirve yanılsamasını açıkça \
yaz; DİP ALTI ise düşük kazancın kalıcı sanılmasına karşı uyar; \
'normalize_taban_deger'i EPV ile birlikte taban çıpası olarak an (kaynak \
SEC 10 yıllık seriyse söyle — bu Yahoo'nun 4 yılından çok daha güvenilir). \
(b) 'ileri_tahmin_ve_bantlar': forward EPS düşük-yüksek aralığı GENİŞSE \
('belirsizlik' YÜKSEK) tek forward F/K'ya yaslanma, aralığı ver; \
'fk_tarihsel_bant' konumunu yorumla — bandın üstü çoğu zaman ÇARPAN \
GENİŞLEMESİDİR (duygu), temel iyileşmesi değil; ikisini ayır. \
(c) 'onceki_analiz_kayitlari' varsa KALİBRASYON dürüstlüğü yap: önceki \
hükümle bugünkünü kıyasla; hüküm değiştiyse NEDENİNİ tek cümleyle söyle \
(fiyat mı hareket etti, veri mi değişti, tez mi kırıldı); fiyat önceki \
hükmün yönünü doğruladıysa ya da yalanladıysa bunu açıkça not et — geçmiş \
hükmü savunmak için bugünü eğme.

15) KADEMELİ ÇIKIŞ / SATIŞ PLANI (zorunlu bölüm): 'cikis_plani' iskeletini \
rafine et — kademeli KÂR ALMA seviyelerini değer çıpalarıyla gerekçele \
(hangi seviye neden mantıklı), TEZ-BOZAN tetikleri kendi falsifikasyon \
listenle BİRLEŞTİR (fiyattan bağımsız çıkış > fiyat hedefi; en güçlü satış \
kuralı tezin kırılmasıdır), zaman stopunu katalizör haritasına bağla. \
'pozisyon' doluysa maliyet-çıpası (disposition effect) tuzağına karşı açık \
uyarı ver: satış kararı giriş fiyatına değil DEĞERE bağlanır. Skor PAHALI \
bölgedeyse elde tutanın kâr alma disiplinini net yaz. Plan TAVSİYE \
DEĞİLDİR; bunu belirt.

16) TUTMA UFKU (kritik — kullanıcının gerçek sorusu): Pakette 'tutma_ufku' \
var; kullanıcının planladığı süre budur (ör. 3-4 ay). Analizi bu ufka BAĞLA: \
(a) kısa ufukta DCF adil değere yürüyüşün TAMAMLANMAYABİLECEĞİNİ açıkça yaz; \
(b) ufka düşen bilanço/olayları tek tek işaretle ve her biri için 'tezi \
ilerletir / bozar' kriterini yaz; (c) tipik ±1σ bandı ve pencere-içi çekilme \
istatistiğiyle 'normal gürültü' ile 'tez kırılması'nı AYIR; (d) zaman stopu \
tarihini net yaz ve o gün için karar kuralı koy. YÖN TAHMİNİ YAPMA — 'daha \
düşer mi?' sorusuna kimse cevap veremez; senin işin aralık, takvim ve karar \
kuralıdır. Bu sınırı kullanıcıya açıkça söyle.

17) STOP & DESTEK (zorunlu): 'stop_ve_destek' paketini kullan — giriş \
dilimlerini ve kâr-alma hedeflerini destek/direnç BÖLGELERİYLE çapraz oku; \
Chandelier trailing ve giriş stopunu tez-bozan tetiklerle BİRLEŞTİR; \
pozisyon boyutunu risk kuralıyla ilişkilendir. Uyarıları aynen taşı: stop \
yön tahmini değildir, gap'te fiyat garantisi yoktur, ortalamaya-dönüş \
rejiminde stop beklenen getiriyi düşürebilir (Kaminski-Lo). Fibonacci/\
pivot ÖNERME (kanıt: folklor). 'veri_tazeligi' uyarısı varsa raporda öne al.

18) KANIT ETİKETİ ve GÜVEN KALİBRASYONU (zorunlu): Önemli iddiaları \
kaynağına göre etiketle — [VERİ] paketteki sayı, [ARAMA] web bulgusu \
(tarih + kaynak adıyla), [YORUM] kendi çıkarımın. İçeriden işlemlerde \
kaynağı AÇIKÇA yaz ('akis_ve_katalizor.iceriden_islemler.ozet.kaynak' alanı: SEC Form 4 \
birincil mi, Yahoo yedeği mi); 'alım yok' ifadesini tek başına ayı \
sinyali sayma — \
açık piyasa alımı nadirdir ve 10b5-1 planlı satışlar sinyal gücü taşımaz \
(planli_satis_adet alanına bak). Rapor sonunda: (a) pakette BOŞ/None \
gelen KRİTİK alanları listele (veri dürüstlüğü), (b) hükmüne 0-10 arası \
GÜVEN NOTU ver, (c) güveni en çok düşüren TEK varsayımı ('en zayıf \
halka') bir cümleyle yaz.

18-B) HABER & BEKLENTİ KATMANI (zorunlu, 'haber_ve_olaylar' paketi): \
(a) 'ima_edilen_hareket' varsa KARAR ÖZETİ KARTINA "Kazanç ima edilen \
hareket: ±%X (vade …)" satırı ekle; bunun ~1σ beklenti olduğunu, GARANTİ \
olmadığını ve piyasanın hareketleri sistematik ABARTTIĞINI (IV crush) \
belirt. Alım/kâr-alma seviyelerini bu bantla çapraz oku (ör. giriş dilimi \
ima edilen alt bandın içinde mi?). (b) 'haber_ve_olaylar.sec_8k_olaylar.kirmizi_bayrak' \
doluysa bunları TEZ-BOZAN riskler bölümünde ÖNE al (Item 4.02/4.01/3.01/ \
1.05/1.03) ve her birini 8-K metninden doğrulaman gerektiğini yaz. (c) \
'haber_ve_olaylar.haber_akisi' başlıklarını [ARAMA] değil [VERİ-HABER] \
etiketiyle kullan; \
haber tabanlı kısa vadeli sinyalin bireysel yatırımcı için hız \
dezavantajı taşıdığını, bu katmanın "sürpriz olmamak" için olduğunu bir \
cümleyle belirt. Hiçbir haberden yön TAHMİNİ üretme.

18-C) SEVİYE SENARYOLARI ve GEÇMİŞ DENETİM: 'seviye_senaryolari' paketi \
"kapanış X seviyesinin altında/üstünde KALIRSA tarihsel olarak ne oldu" \
baz oranlarını verir — bunları "şu seviyenin altında günlük kapanış \
kalırsa..." cümleleriyle karar matrisine bağla; N<8 olan satırdan SONUÇ \
ÇIKARMA, N'i açıkça yaz. Bunlar TAHMİN değil KOŞULLU BAZ ORANDIR.

18-D) REJİM & ZAMANLAMA (rapor G4/G5/G10): 'rejim_ve_zamanlama' paketi \
(12-1 momentum, PEAD, MA200) yön tahmini DEĞİL tempo bilgisidir. Momentum \
veya PEAD aleyhteyse kademeli girişin SEVİYELERİNİ değil TEMPOSUNU \
yavaşlatmayı öner ve bunu açıkça yaz; lehteyse 'destekleyici rüzgâr' de \
ama kesinlik iddia etme. PEAD penceresi ~60 işlem günüdür ve etki \
büyük-cap'te zayıftır — bunu belirt. Bu sinyallerden yön tahmini üretme.

19) YATIRIM KOMİTESİ (IC) KARAR DİSİPLİNİ (zorunlu — profesyonel analist \
memosu standardı): \
(a) Raporun EN BAŞINA "KARAR ÖZETİ KARTI" koy — 8 satırlık markdown tablo: \
Hüküm (aday/izle/pas + koşul) · Güven (0-10) · Olasılık-ağırlıklı beklenen \
getiri (12 ay VE kullanıcının tutma ufku için AYRI) · Risk/Ödül asimetrisi \
(yukarı yol ÷ aşağı yol) · Alım bölgesi koşulu · Stop (giriş + trailing, \
stop_ve_destek paketinden) · Zaman stopu tarihi · Bir sonraki kontrol \
tarihi. Kartın altına tek satır: "Nihai karar ve sorumluluk kullanıcıya \
aittir; bu bir model hükmüdür, emir değildir." \
(b) BEKLENEN DEĞER TABLOSU: Ayı/Baz/Boğa satırları — her birinde olasılık \
(toplam %100), 12 aylık hedef bölge, kullanıcı ufkunda (tutma_ufku) \
GERÇEKÇİ hedef ve dayandığı tek cümlelik varsayım. Kendi olasılıklarını \
motorun 'senaryo_beklenen_deger' ve 'monte_carlo_adil_deger' çıktısıyla \
UZLAŞTIR; ayrışıyorsan nedenini bir cümleyle yaz. \
(c) KARAR MATRİSİ (en az 3 satır): "fiyat ŞU bölgeye gelirse VE ŞU \
ölçülebilir koşul sağlanırsa → çerçeve eylemi (izlemeye al / alım bölgesi \
adayı / dilim tamamla / eksilt / pas)". Koşullar ölçülebilir olsun \
(rakam, tarih, olay) — "iyileşirse" gibi belirsiz laf YASAK. \
(d) KARŞI TARAF SORGUSU: "Bu fiyattan bana satan kim ve neden haklı \
olabilir?" — 2-3 cümle; en güçlü satıcı gerekçesini ciddiye al. \
(e) TARİHLİ İZLEME PLANI: katalizör paketindeki tarihleri kullan; her \
tarih için "tezi İLERLETİR işaret" ve "tezi BOZAR işaret" yaz. \
(f) Dil: AL/SAT emri verme; profesyonel memo dili kullan (aday/izle/pas + \
koşul). Kesinlik iddiası yasak — kart model hükmüdür.

20) 5-SORU ENTEGRASYONU: Pakette 'nitel_5_soru_cevaplari' varsa \
Anlaşılırlık notunu (X/5) KARAR ÖZETİ KARTINA ek satır olarak taşı; X<4 \
ise hükmü otomatik TEMKİNLİ çerçevele ve nedenini açıkça yaz ("şirket \
yeterince anlaşılmadan güven yükseltilmez"). Alan yoksa Veri Sınırları \
bölümünde "5 Soru turu çalıştırılmamış" diye belirt.

21) EYLEM DİSİPLİNİ — "NE YAPMALI" SÖZLEŞMESİ (zorunlu, tüm bölümler): \
(a) HER ana bölümün SON satırı tam olarak şu kalıpla biter: "→ NE \
YAPMALI: Eğer [ölçülebilir eşik/tarih/olay] ise → [somut adım]." Eşik ve \
adımlar YALNIZ paketteki sayılardan (kademeli_plan, cikis_plani, \
stop_ve_destek, seviye_senaryolari, katalizor_takvimi) türetilir; pakette \
eşik yoksa kalıp şudur: "→ NE YAPMALI: İzle — [metrik], kontrol: [tarih]". \
Fiyat/eşik UYDURMAK yasaktır. (b) TEZİ ÇÜRÜTECEK 3 GÖSTERGE (Falsifikasyon \
bölümünün sonunda, zorunlu tablo): gösterge | ölçülebilir eşik | \
tetiklenirse eylem. "Kötüleşirse" gibi ölçüsüz ifade YASAK. (c) İZLEME \
LİSTESİ zorunlu tablo biçimindedir: metrik | eşik | aşılırsa eylem. \
(d) KATALİZÖR satırları dört sütundur: tarih | beklenti | ÖNCESİNDE eylem \
| SONRASINDA eylem (ör. "önce: yeni dilim yok", "sonra: rehber düşerse \
eksilt adayı"). (e) KARAR GÜNLÜĞÜ & KALİBRASYON (kapanışta tek paragraf): \
kullanıcıya, Beklenen Değer Tablosundaki olasılıkları uygulamanın Kayıt \
Defteri'ne yazmasını ve ufuk dolunca Brier skoruyla puanlanacağını \
hatırlat — tahmin disiplini böyle kurulur. (f) DİL SINIRI: "al / sat / \
şimdi gir / senin için uygun / garanti / kesin" kalıpları YASAK; daima \
koşullu ve kişiye-özel-olmayan dil kullan ("eğer X ise plandaki Y \
tetiklenir; nihai karar kullanıcınındır"). (g) UYUM DENETİMİ (raporu \
bitirmeden): tüm zorunlu bölümler, üç tablo ((b), (c) ve Beklenen Değer) \
ve her ana bölümde tam BİR "→ NE YAPMALI" satırı var mı — eksik varsa \
ilgili bölümü TAMAMLAyıp öyle bitir.

FORMAT: Türkçe, markdown. HER ana bölümün SON satırı Talimat 21(a) \
kalıbındaki tek bir "→ NE YAPMALI:" cümlesidir. Bölümler: **KARAR ÖZETİ KARTI (tablo)** → \
**HÜKÜM (fiyat ekseni)** → **Yatırım Tezi \
ve Variant Perception** (tek cümle tez + piyasa neyi kaçırıyor + 3-5 izlenebilir \
öncül) → **İki Eksen: Şirket Kalitesi vs Fiyat** (DuPont, sermaye tahsisi ve \
moat dahil) → **Seçtiğim Derinleşme Ekseni** → **Değerleme Sorgusu (Ters-DCF, EPV \
tabanı, Monte Carlo ve senaryo beklenen değeri dahil)** → **Rakip Kıyası ve \
Sektör Merceği** → \
**Çelişki Dedektörü Bulguları** → **Falsifikasyon, Pre-mortem ve En Kötü \
Senaryo + Tezi Çürütecek 3 Gösterge (tablo: gösterge|eşik|eylem)** → **Boğa vs Ayı (olasılıklı)** → **Beklenen Değer Tablosu ve Karar \
Matrisi** → **Karşı Taraf Sorgusu** → **Tarihli İzleme Planı (her tarihte ÖNCE/SONRA eylemli)** → \
**Alım Bölgesi Koşulları ve İzleme \
Listesi (tablo: metrik|eşik|aşılırsa eylem)** (tavsiye değil: hangi fiyat/koşul gerçekleşirse tez değişir) → \
**Kademeli Giriş / Bekleme Planı (disiplin çerçevesi)** → **Kademeli Çıkış / \
Satış Planı (disiplin çerçevesi)** → **Güven Kalibrasyonu (0-10) ve En \
Zayıf Halka** → **Veri Sınırları ve Boş Alanlar**. \
BÖLÜM ÖNCELİĞİ (uzunluk çatışmasını çözer): Yukarıdaki liste ~18 \
bölümdür; 1300-1900 kelimeye sıkıştırılırsa bölüm başına ~80 kelime \
düşer ve HEPSİ yüzeysel kalır. Bu yüzden bölümler İKİ SINIFTIR: \
(A) DERİNLEŞTİRİLECEK — Karar Özeti Kartı, Hüküm, Yatırım Tezi, \
Değerleme Sorgusu, Falsifikasyon/Pre-mortem, Beklenen Değer Tablosu \
ve Karar Matrisi, Güven Kalibrasyonu. Bunlara toplam kelimenin en az \
%70'ini ayır. (B) KISA GEÇİLEBİLİR — kalan bölümler; ilgili veri yoksa \
TEK SATIRDA 'bu koşuda uygulanamaz, sebep: …' yaz ve geç. Bir bölümü \
atlamak yerine kısa geçmek ESASTIR; ama A sınıfını kısaltıp B sınıfını \
şişirme. Hedef 1800-2600 kelime (A sınıfı derinleşince alt sınır yükselir). Kapanışta iki satır: Talimat 21(e) karar-günlüğü hatırlatması + "Bu \
analiz yatırım tavsiyesi değildir." """


# ── B10-10: finansal analizde kaynak kalitesi için alan adı beyaz listesi.
# Boş bırakılırsa tüm web taranır; dolu olduğunda Claude YALNIZ bu alanlarda
# arar. SEC birincildir; kalanlar yaygın finans kaynaklarıdır.
WEB_ARAMA_ALANLARI = [
    "sec.gov", "investor.gov", "federalreserve.gov", "bls.gov",
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "barrons.com", "marketwatch.com", "seekingalpha.com",
    "fool.com", "morningstar.com", "finance.yahoo.com",
]


def _api_hata_mesaji(e, api_key=""):
    """B10-11: istisnayı kullanıcıya gösterilebilir hale getir + ANAHTARI SİL.

    Traceback ya da hata mesajı API anahtarını taşıyabilir; ekrana basmadan
    önce maskelenir.
    """
    msg = f"{type(e).__name__}: {e}"
    if api_key and len(api_key) > 12:
        msg = msg.replace(api_key, "sk-ant-***MASKELENDI***")
    return msg[:400]


def _claude_cagir(client, kwargs, api_key="", tekrar=3):
    """B10-2: TİPLİ istisna işleme.

    Eski sürüm tek bir `except Exception` ile HER hatayı "web araması
    çalışmadı" sanıyor ve aracı çıkarıp TEKRAR deniyordu. Bu, kimlik
    doğrulama hatasında, geçersiz model kimliğinde ve bağlam taşmasında
    ikinci bir başarısız çağrı yakıp kullanıcıya YANLIŞ teşhis veriyordu.
    Artık hata sınıfına göre davranılır:
      401 kimlik      → tekrar YOK, kullanıcıdan anahtar iste
      404 model       → tekrar YOK, model kimliği geçersiz
      400 istek       → tekrar YOK (bağlam taşması/araç kapalı burada ayrışır)
      429 hız sınırı  → üstel bekleme + jitter
      529 aşırı yük   → üstel bekleme + jitter
      ağ hatası       → sınırlı tekrar
    Dönen: (yanit, arac_kapatilmali_mi, kullanici_mesaji)
    """
    import time
    import random
    try:
        import anthropic as _an
    except Exception:
        _an = None
    son = None
    for deneme in range(tekrar):
        try:
            return client.messages.create(**kwargs), False, None
        except Exception as e:
            son = e
            ad = type(e).__name__
            if _an is not None:
                if isinstance(e, getattr(_an, "AuthenticationError", ())):
                    raise RuntimeError(
                        "API anahtarı geçersiz veya süresi dolmuş (401). "
                        "Kenar çubuğundan yeni anahtar gir. "
                        "TEKRAR DENENMEDİ — anahtar değişmeden sonuç "
                        "değişmez.") from e
                if isinstance(e, getattr(_an, "NotFoundError", ())):
                    raise RuntimeError(
                        f"Model kimliği bulunamadı (404): '{kwargs.get('model')}'. "
                        f"Model emekli edilmiş olabilir — 'Hesabımdaki "
                        f"modelleri listele' ile güncel listeyi çek.") from e
                if isinstance(e, getattr(_an, "BadRequestError", ())):
                    _m = str(e).lower()
                    if "web search" in _m or "tool" in _m:
                        return None, True, (
                            "Web arama aracı hesapta AÇIK DEĞİL (400). "
                            "Konsol → Settings'ten etkinleştirebilirsin; "
                            "bu koşu araçsız üretildi.")
                    raise RuntimeError(
                        "İstek reddedildi (400). En olası sebep BAĞLAM "
                        "TAŞMASI (veri paketi çok büyük) ya da geçersiz "
                        "parametre. Ayrıntı: "
                        + _api_hata_mesaji(e, api_key)) from e
                if isinstance(e, (getattr(_an, "RateLimitError", ()),
                                  getattr(_an, "InternalServerError", ()))):
                    if deneme < tekrar - 1:
                        time.sleep(2 ** deneme + random.uniform(0, 0.5))
                        continue
                    raise RuntimeError(
                        "Sunucu meşgul / hız sınırı (429-529) ve tekrarlar "
                        "tükendi. Birkaç dakika sonra dene.") from e
                if isinstance(e, getattr(_an, "APIConnectionError", ())):
                    if deneme < tekrar - 1:
                        time.sleep(1.5 * (deneme + 1))
                        continue
                    raise RuntimeError(
                        "Ağ bağlantısı kurulamadı. İnternet/proxy kontrol "
                        "et.") from e
            if deneme < tekrar - 1 and ad in ("APIStatusError", "APIError"):
                time.sleep(1.5 * (deneme + 1))
                continue
            raise
    raise son


def _metin_ve_alintilar(resp):
    """B10-1: metni VE ALINTILARI çıkar.

    ESKİ KOD yalnız `type == "text"` bloklarını alıyordu. Metin kaybolmuyor
    (cevap zaten text bloklarında) AMA her `citations` dizisi SESSİZCE
    ATILIYORDU. Anthropic'in web arama dokümantasyonu açık: çıktı doğrudan
    son kullanıcıya gösteriliyorsa ALINTILAR DA GÖSTERİLMELİDİR. Finansal
    bir araçta bu hem uyum hem güven meselesidir — ayrıca alıntı alanları
    (url/title/cited_text) TOKEN SAYMAZ, yani göstermek bedavadır.
    """
    parcalar, alintilar, arama_sayisi = [], [], 0
    for b in getattr(resp, "content", []) or []:
        t = getattr(b, "type", None)
        if t == "text":
            parcalar.append(getattr(b, "text", "") or "")
            for c in (getattr(b, "citations", None) or []):
                try:
                    alintilar.append({
                        "url": getattr(c, "url", None),
                        "baslik": getattr(c, "title", None),
                        "alinti": (getattr(c, "cited_text", None) or "")[:200]})
                except Exception:
                    continue
        elif t == "server_tool_use":
            arama_sayisi += 1
    # tekilleştir (aynı kaynak birden çok cümlede anılabilir)
    gorulen, tekil = set(), []
    for a in alintilar:
        k = (a.get("url"), a.get("alinti"))
        if k in gorulen:
            continue
        gorulen.add(k)
        tekil.append(a)
    return "\n".join(parcalar), tekil, arama_sayisi


def _kullanim_ozeti(resp, arama_sayisi):
    """B10-6: web arama ve token maliyetini KULLANICIYA göster.

    Anthropic web aramasını 1.000 arama başına 10 $ ücretlendirir
    (arama başına ~0.01 $) ve getirilen sayfa içeriği AYRICA girdi tokeni
    olarak sayılır. Kendi anahtarıyla ödeyen kullanıcı bunu görmelidir.
    """
    u = getattr(resp, "usage", None)
    gir = getattr(u, "input_tokens", None) if u else None
    cik = getattr(u, "output_tokens", None) if u else None
    onb_o = getattr(u, "cache_read_input_tokens", None) if u else None
    onb_y = getattr(u, "cache_creation_input_tokens", None) if u else None
    stu = getattr(u, "server_tool_use", None) if u else None
    ara = getattr(stu, "web_search_requests", None) if stu else None
    if ara is None:
        ara = arama_sayisi
    parca = []
    if gir is not None:
        parca.append(f"girdi {gir:,} token")
    if cik is not None:
        parca.append(f"çıktı {cik:,} token")
    if onb_o:
        parca.append(f"önbellekten okunan {onb_o:,} token (~%90 ucuz)")
    if onb_y:
        parca.append(f"önbelleğe yazılan {onb_y:,} token")
    if ara:
        parca.append(f"{ara} web araması (~${ara * 0.01:.2f})")
    return " · ".join(parca) if parca else None


def call_claude(api_key, model, payload, use_web=True, max_tokens=8000):
    """Claude API çağrısı — alıntılı, tipli hata işlemeli, önbellekli."""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # B10-3: modele HANGİ ALANLARIN GERÇEKTEN OLDUĞUNU söyle.
    # Araştırma bulgusu: bir talimat pakette OLMAYAN bir alanı adıyla
    # isterse model çoğunlukla DURUMU BİLDİRMEZ, UYDURUR (FaithEval:
    # bağlam yokken doğru "bilmiyorum" oranı bir modelde %76.8'den
    # %7.4'e düşüyor). Bu yüzden mevcut/eksik anahtar listesi ENJEKTE
    # edilir — böylece uydurma yerine "veri yok" demesi kolaylaşır.
    _sozlesme = payload_sozlesmesi(payload)

    user_msg = (
        f"Ticker: {payload['meta']['ticker']} — {payload['meta']['sirket']}\n"
        f"Aşağıda Python motorunun ürettiği veri paketi var. Sayısal skor bir "
        f"sezgiseldir; körü körüne güvenme, denetle.\n\n"
        f"{_sozlesme}\n\n"
        # B10-5: indent=1 biçimlendirme her anahtarda satır sonu + boşluk
        # ekliyordu (ölçüm: minimal pakette bile %18.6 israf; dolu pakette
        # mutlak tasarruf çok daha büyük). Model için okunabilirlik farkı
        # YOK, maliyet farkı VAR.
        f"```json\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        f"```"
    )

    # B10-4: ÖNBELLEK. Sistem talimatı ~4.600 token ve HER çağrıda AYNI.
    # cache_control ile ilk çağrıdan sonra bu önek taban fiyatın 0.1 katına
    # okunur (Anthropic: uzun promptlarda maliyette %90'a, gecikmede %85'e
    # varan azalma). DİKKAT: tools ya da model değişirse önbellek geçersiz
    # olur — araçsız geri düşüş yolu bu yüzden her zaman ıskalar.
    sistem = [{"type": "text", "text": CLAUDE_SYSTEM,
               "cache_control": {"type": "ephemeral"}}]
    kwargs = dict(model=model, max_tokens=max_tokens, system=sistem,
                  messages=[{"role": "user", "content": user_msg}])

    note = ""
    if use_web:
        arac = {"type": "web_search_20250305", "name": "web_search",
                "max_uses": 6}
        if WEB_ARAMA_ALANLARI:
            arac["allowed_domains"] = list(WEB_ARAMA_ALANLARI)
        kwargs["tools"] = [arac]

    try:
        resp, arac_kapat, uyari = _claude_cagir(client, kwargs, api_key)
    except RuntimeError:
        raise                                  # zaten kullanıcıya uygun
    if arac_kapat:
        kwargs.pop("tools", None)
        resp, _, _ = _claude_cagir(client, kwargs, api_key)
        note = f"\n\n---\n> ⚠️ {uyari}"

    text, alintilar, ara_say = _metin_ve_alintilar(resp)
    kul = _kullanim_ozeti(resp, ara_say)

    if alintilar:
        note += ("\n\n---\n### 🔗 Web Kaynakları (Claude'un aradığı)\n"
                 + "\n".join(
                     f"{i+1}. [{a['baslik'] or a['url']}]({a['url']})"
                     + (f"\n   > {a['alinti']}" if a.get("alinti") else "")
                     for i, a in enumerate(alintilar[:20]))
                 + "\n\n*Alıntılar modelin gerçekten getirdiği sayfalardan; "
                   "bir iddiayı doğrulamak için alıntı metniyle iddiayı "
                   "KARŞILAŞTIR — atıf ile içerik uyuşmazlığı bilinen bir "
                   "risktir.*")
    elif use_web and not arac_kapat:
        note += ("\n\n---\n> ℹ️ Web araması açıktı ama model hiç arama "
                 "yapmadı ya da alıntı döndürmedi — rapordaki web "
                 "iddialarını [HİPOTEZ] say.")
    if kul:
        note += f"\n\n<sub>Kullanım: {kul}</sub>"
    return (text.strip() or "_Modelden metin dönmedi._") + note


CLAUDE_NITEL_SYSTEM = (
    "Sen, yatırımcıya bir şirketi ANLATAN kıdemli bir analistsin. "
    "Okuyucun finans jargonu bilmiyor ve 10-K okumuyor; senin işin o "
    "metnin cevapladığı soruları onun yerine cevaplamak.\n\n"
    "KURALLAR (hepsi zorunlu):\n"
    "• Dil: sade Türkçe, lise seviyesi. 'Moat, EBITDA, sinerji, "
    "vertikal entegrasyon' gibi terimler YASAK; kullanman gerekirse "
    "parantez içinde tek cümleyle açıkla.\n"
    "• Her sorunun cevabı 2-4 cümle. Uzun yazma; anlaşılır yaz.\n"
    "• Her iddiayı etiketle: [ARAMA] webden bulduğun (kaynak adı + "
    "tarih), [VERİ] gönderilen pakette olan sayı, [YORUM] senin "
    "çıkarımın.\n"
    "• Doğrulayamadığın yere AÇIKÇA 'BİLİNMİYOR' yaz. Uydurma, "
    "tahmini gerçek gibi sunma. Bilinmeyeni işaretlemek, yanlış "
    "doldurmaktan İYİDİR.\n"
    "• 4. soruda somut ol: geçen yıl açıklanan hedef/rehber ne idi, "
    "gerçekleşen ne oldu; bulamazsan 'BİLİNMİYOR' yaz.\n"
    "• 3. soruda gerçek tehditleri yaz, genel geçer laf etme "
    "('rekabet artabilir' gibi boş cümle YASAK).\n\n"
    "ÇIKTI FORMATI (markdown, Türkçe):\n"
    "### Tek Cümlede Şirket\n(bir cümle, jargonsuz)\n\n"
    "Sonra 5 soruyu sırayla başlık yapıp cevapla.\n\n"
    "### Anlaşılırlık Notu\n"
    "Kaç soruya NET ve kaynaklı cevap verebildin: X/5. Ardından tek "
    "cümlelik karar kuralı: 4'ten azsa 'bu şirket senin için yeterince "
    "anlaşılır değil — pas geçmek meşru ve çoğu zaman doğru "
    "hamledir' de. Not: bu bir kalite ölçüsü DEĞİL, anlaşılırlık "
    "ölçüsüdür; anlaşılır şirket iyi yatırım demek değildir.\n\n"
    "### En Az Emin Olduğum Nokta\n(tek cümle)\n\n"
    "Kapanış: 'Bu analiz yatırım tavsiyesi değildir.'"
)


NITEL_SORULAR = (
    "1) Bu şirket tam olarak NE satıyor, KİME satıyor?",
    "2) Müşteriler neden bu firmadan alıyor, rakibinden değil? "
    "Bu sebep 3 yıl sonra da geçerli mi?",
    "3) Bu şirketi 3 yılda NE ÖLDÜRÜR? (rakip, regülasyon, teknoloji, "
    "tek müşteriye bağımlılık)",
    "4) Yönetim geçen yıl NE SÖZ VERDİ, tuttu mu?",
    "5) Kâr NEREDEN geliyor ve tekrarlanabilir mi?",
)


def call_claude_nitel(api_key, model, payload, use_web=True,
                      max_tokens=3000):
    """
    5 SORU — ŞİRKETİ ANLAT (nitel katman).
    Sayısal motorun cevaplayamadığı, 10-K metnine ait soruları Claude
    araştırıp SADE Türkçe cevaplar. Kurallar sistem talimatında sert:
    teknik terim yok, uydurma yok, doğrulanamayan yerde 'BİLİNMİYOR'.
    Sonda 5 üzerinden 'anlaşılırlık' notu ve karar kuralı verilir —
    net cevap sayısı düşükse doğru hamle 'pas geçmek'tir.
    """
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    sysm = CLAUDE_NITEL_SYSTEM
    meta = payload.get("meta", {})
    um = (f"Şirket: {meta.get('ticker')} — {meta.get('sirket')} "
          f"({meta.get('sektor')} / {meta.get('endustri')})\n\n"
          "Aşağıdaki 5 soruyu cevapla:\n"
          + "\n".join(NITEL_SORULAR)
          + "\n\nSayısal veri paketi (bağlam için; nitel soruları "
            "cevaplarken web aramasını kullan):\n```json\n"
          + json.dumps({k: payload.get(k) for k in
                        ("meta", "buyume_ve_marjlar", "kalite_ve_adli",
                         "rakip_kiyasi", "akis_ve_katalizor",
                         "sektor_sablonu", "kazanc_cagrisi")
                        if payload.get(k) is not None},
                       ensure_ascii=False)[:14000] + "\n```")
    kwargs = dict(model=model, max_tokens=max_tokens, system=sysm,
                  messages=[{"role": "user", "content": um}])
    if use_web:
        kwargs["tools"] = [{"type": "web_search_20250305",
                            "name": "web_search", "max_uses": 6}]
    try:
        resp = client.messages.create(**kwargs)
    except Exception:
        kwargs.pop("tools", None)
        resp = client.messages.create(**kwargs)
    text = "\n".join(b.text for b in resp.content
                      if getattr(b, "type", None) == "text")
    return text.strip() or "_Modelden metin dönmedi._"


MANUEL_BASLIK = (
    "Aşağıdaki TALİMAT bloğuna göre, en sonda verilen VERİ PAKETİNİ "
    "analiz et. Talimatta istenen bölümleri eksiksiz üret. Paket "
    "dışındaki bilgiler için (ürün, rakip, yönetim sözleri, haberler) "
    "web araması yap; doğrulayamadığına 'BİLİNMİYOR' yaz, uydurma.\n\n"
    "— — — TALİMAT — — —\n")


def _payload_kisalt(p, limit=90000):
    """Manuel modda yapıştırma kolaylığı için hacimli alanları kırp.
    Kırpılan alan SESSİZCE atılmaz; listesi istemin içine yazılır."""
    import copy
    q = copy.deepcopy(p)
    atilan = []
    agir = ["onceki_analiz_kayitlari", "baz_oranlar", "sektor_sablonu",
            "ileri_tahmin_ve_bantlar", "nakit_donusum_dongusu",
            "gelir_tablosu_musd", "nakit_akisi_musd", "bilanco"]
    for k in agir:
        if len(json.dumps(q, ensure_ascii=False)) <= limit:
            break
        if q.get(k) is not None:
            q.pop(k, None)
            atilan.append(k)
    if atilan:
        q["_kirpilan_alanlar"] = atilan
        q["_kirpma_notu"] = ("Bu alanlar uzunluk sınırı için çıkarıldı; "
                             "gerekiyorsa kullanıcıdan iste.")
    return q


def manuel_istem(m, tur="derin", kisa=True):
    """
    API'SİZ KULLANIM: claude.ai sohbetine yapıştırılacak TAM istemi üretir.
    Abonelik (Pro/Max) ile API ayrı ürünlerdir; abonelik programa API
    erişimi vermez. Bu yol, aynı analizi ek ücret ödemeden almanı sağlar:
    program istemi hazırlar, sen sohbete yapıştırır/dosya olarak eklersin.
    Tek kayıp: otomasyon (rapor sohbette kalır, uygulamaya geri yazılmaz).
    """
    p = build_payload(m)
    if tur == "5soru":
        talimat = CLAUDE_NITEL_SYSTEM + "\n\nCEVAPLANACAK SORULAR:\n" + \
            "\n".join(NITEL_SORULAR)
        p = {k: p.get(k) for k in ("meta", "buyume_ve_marjlar",
                                   "kalite_ve_adli", "rakip_kiyasi",
                                   "akis_ve_katalizor", "kazanc_cagrisi")
             if p.get(k) is not None}
    else:
        talimat = CLAUDE_SYSTEM
        if kisa:
            p = _payload_kisalt(p)
    # B10-3: API yolunda paket sözleşmesi enjekte ediliyordu ama MANUEL
    # yolda edilmiyordu — aynı uydurma riski (talimat olmayan bir alanı
    # adıyla isterse model doldurmaya çalışır). İki yol da aynı korumayı
    # almalı.
    try:
        _soz = "\n\n" + payload_sozlesmesi(p) if tur != "5soru" else ""
    except Exception:
        _soz = ""
    return (MANUEL_BASLIK + talimat + _soz
            + "\n\n— — — VERİ PAKETİ (" + str(m.get("ticker")) + ") — — —\n"
            + "```json\n"
            # B10-5b: manuel mod da MODELE gidiyor (kullanıcı yapıştırıyor)
            # ve kodun kendi uzunluk sınırı var (_payload_kisalt alan
            # atıyor). Compact biçim, alan atılmadan önce DAHA ÇOK
            # içeriğin sığmasını sağlar.
            + json.dumps(p, ensure_ascii=False,
                         separators=(",", ":")) + "\n```")


def call_claude_redteam(api_key, model, payload, rapor, max_tokens=2500):
    """
    İKİNCİ TUR — KIRMIZI TAKIM: ilk raporu ÇÜRÜTMEYE çalışır. Amaç
    doğrulama değil yıkım denemesidir; ayakta kalan tez güçlü tezdir.
    Aynı model, ayrı sistem talimatı; ek maliyet ~1 çağrı.
    """
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    # ── B10-9 ────────────────────────────────────────────────────────
    # Kanıt durumu: MODELİN KENDİ ÇIKTISINI dış geri bildirim olmadan
    # düzeltmesi güvenilir DEĞİLDİR (Huang vd., ICLR 2024: içsel öz-düzeltme
    # bazen performansı DÜŞÜRÜYOR). Yardımcı olduğu durum, eleştirinin
    # DIŞARIDAN DOĞRULANABİLİR olmasıdır. Bu yüzden burada her itiraz için
    # veri paketinden ALAN ADI + DEĞER zorunlu kılınıyor; dayanağı
    # gösterilemeyen itiraz "SPEKÜLATİF" diye etiketlenmek zorunda.
    # Ayrıca aynı modelin kendini eleştirmesi KORELE KÖR NOKTA taşır —
    # bu, çıktının başına açıkça yazılır.
    sysm = ("Kıdemli bir 'kırmızı takım' hisse analistisin. Görevin: sana "
            "verilen raporu KANITA DAYANARAK çürütmeye çalışmak — nazik "
            "olma, haksız da olma.\n\n"
            "ZORUNLU KANIT KURALI: Her itiraz için veri paketinden "
            "DAYANAK göster — 'alan_adı = değer' biçiminde. Dayanağını "
            "pakette gösteremiyorsan o itirazı [SPEKÜLATİF] diye "
            "etiketle. Dayanaksız itiraz üretmek, itiraz üretmemekten "
            "KÖTÜDÜR: okuyucunun zamanını çalar ve gerçek riski gizler.\n\n"
            "Çıktı (Türkçe, ≤600 kelime):\n"
            "0) İLK SATIR aynen şu olacak: '⚠️ Bu eleştiri, raporu YAZAN "
            "modelin kendisi tarafından üretildi — korele kör nokta "
            "taşır ve bağımsız doğrulama DEĞİLDİR.'\n"
            "1) En güçlü karşı-tez (en fazla 3 madde; her biri "
            "'dayanak: alan_adı = değer' ile).\n"
            "2) Raporun paketle ÇELİŞEN cümleleri: cümleyi kısaca alıntıla "
            "VE çeliştiği alanı göster. Çelişki bulamadıysan 'çelişki "
            "bulunamadı' yaz — uydurma.\n"
            "3) Aşırı güven noktaları ve atlanan riskler.\n"
            "4) SONUÇ: hüküm bu eleştiriden sonra AYAKTA / YARA ALDI / "
            "ÇÖKTÜ + tek cümle gerekçe. Bu hüküm bir OYLAMA değil, "
            "okuyucuya bir SORU LİSTESİDİR.\n"
            "Kapanışta: 'Bu analiz yatırım tavsiyesi değildir.'")
    um = ("RAPOR:\n" + (rapor or "")[:12000]
          + "\n\nVERİ PAKETİ:\n```json\n"
          + json.dumps(payload, ensure_ascii=False)[:20000] + "\n```")
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  system=sysm,
                                  messages=[{"role": "user",
                                             "content": um}])
    text = "\n".join(b.text for b in resp.content
                     if getattr(b, "type", None) == "text")
    return text.strip() or "_Kırmızı takımdan metin dönmedi._"


def list_account_models(api_key):
    """Hesapta erişilebilir model kimliklerini canlı çek (ezber yerine keşif)."""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    # B10-7: limit=20 varsayılanı sessizce model KAÇIRIR (API 1-1000
    # destekler ve en yeniden eskiye sıralar). Ayrıca sabit model listesi
    # emeklilikte 404 üretir; Anthropic emeklilik için en az 60 gün önceden
    # haber verir ama takip etmek uygulamanın işidir.
    out, tur = [], 0
    try:
        sayfa = client.models.list(limit=100)
        out = [mm.id for mm in sayfa.data]
        while getattr(sayfa, "has_more", False) and tur < 5:
            tur += 1
            sayfa = client.models.list(limit=100,
                                       after_id=out[-1] if out else None)
            out += [mm.id for mm in sayfa.data]
    except TypeError:
        out = [mm.id for mm in client.models.list().data]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# 9) GRAFİKLER (Plotly)
# ---------------------------------------------------------------------------

_PLOTLY_KAYITLI = False


def _plotly_sablon_kaydet():
    """Premium plotly şablonları — kart üstüne oturması için şeffaf
    zemin; Wong (2011) renk-körü-güvenli palet (uygulamanın geneliyle
    tutarlı). NOT: plotly'de font=dict(...) ile font_color aynı çağrıda
    verilemez (ValueError) — renk font dict'inin İÇİNDE verilir."""
    global _PLOTLY_KAYITLI
    if _PLOTLY_KAYITLI:
        return
    import plotly.io as pio
    WONG = ["#56b4e9", "#e69f00", "#009e73", "#f0e442",
            "#0072b2", "#d55e00", "#cc79a7", "#94a3b8"]

    def _yap(txt, grid, zero, hoverbg):
        eksen = dict(gridcolor=grid, zerolinecolor=zero,
                     linecolor=zero, tickcolor=zero)
        return go.layout.Template(layout=go.Layout(
            font=dict(family="Inter, sans-serif", size=13, color=txt),
            title=dict(font=dict(family="Inter, sans-serif",
                                 size=17, color=txt)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=48, r=24, t=48, b=40),
            hoverlabel=dict(bgcolor=hoverbg,
                            bordercolor="rgba(0,0,0,0)",
                            font=dict(family="Inter, sans-serif",
                                      size=12, color=txt)),
            colorway=WONG,
            xaxis=eksen, yaxis=eksen,
            legend=dict(bgcolor="rgba(0,0,0,0)")))

    pio.templates["premium_dark"] = _yap(
        "#e7ecf3", "rgba(255,255,255,0.06)",
        "rgba(255,255,255,0.10)", "#1a2030")
    pio.templates["premium_dark"].data.heatmap = [go.Heatmap(colorscale=[
        [0.0, "#0b0e14"], [0.33, "#153b52"],
        [0.66, "#166a86"], [1.0, "#22c3b8"]])]
    pio.templates["premium_light"] = _yap(
        "#1c1c1e", "rgba(0,0,0,0.06)",
        "rgba(0,0,0,0.12)", "#ffffff")
    _PLOTLY_KAYITLI = True


def _plotly_tema():
    """Grafik şablonu temayı takip eder (premium_dark / premium_light)."""
    try:
        _plotly_sablon_kaydet()
        return ("premium_dark" if _tema_koyu_mu() else "premium_light")
    except Exception:
        return "plotly_white"


_LAYOUT = dict(template=_plotly_tema(), height=360,
               margin=dict(l=40, r=20, t=48, b=40),
               font=dict(family="Inter, Segoe UI, sans-serif", size=12))


def fig_price(hist, ticker):
    df = hist.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    f = go.Figure()
    f.add_scatter(x=df.index, y=df["Close"], name="Fiyat",
                  line=dict(width=1.6, color=WONG["gok"]))
    f.add_scatter(x=df.index, y=df["SMA50"], name="SMA 50",
                  line=dict(width=1, dash="dot", color=WONG["turuncu"]))
    f.add_scatter(x=df.index, y=df["SMA200"], name="SMA 200",
                  line=dict(width=1, dash="dash", color=WONG["mor"]))
    # ── B11-8: LOG ÖLÇEK ─────────────────────────────────────────────
    # Doğrusal ölçekte eşit DOLAR hareketi eşit dikey yer kaplar; ama
    # yatırımcıyı ilgilendiren YÜZDE harekettir. Fiyat pencerede ~2 kat
    # veya fazla hareket ettiyse doğrusal ölçek erken dönemi sıkıştırır
    # ve son dönemi abartır. Finans konvansiyonu: geniş aralıkta log.
    _oran = None
    try:
        _c = df["Close"].dropna()
        if len(_c) > 20 and float(_c.min()) > 0:
            _oran = float(_c.max()) / float(_c.min())
    except Exception:
        _oran = None
    _log = bool(_oran and _oran >= 2.0)
    if _log:
        f.update_yaxes(type="log")
    f.update_layout(
        title=(f"{ticker} — 5 Yıllık Fiyat"
               + (f"  ·  LOG ölçek (aralık {_oran:.1f} kat — doğrusal "
                  f"ölçek erken dönemi sıkıştırırdı)" if _log else "")),
        **_LAYOUT)
    return f


def fig_fundamentals(m):
    ys = m["years"]
    f = go.Figure()
    # B11-4: Material paleti yeşil-kırmızı çiftleri içeriyordu; Wong
    # (Okabe-Ito) paletine geçildi — dosya bu paleti zaten TANIMLIYOR
    # ama grafiklerde HİÇ KULLANMIYORDU (ölçüm: her renk 1 kez = yalnız
    # tanım satırında). Mavi/turuncu/gök ayrımı dötanopide korunuyor.
    for key, label, color in (("rev", "Gelir", WONG["gok"]),
                              ("ni", "Net Kâr", WONG["turuncu"]),
                              ("fcf", "FCF", WONG["mor"])):
        d = m[key]
        f.add_bar(x=[str(y) for y in ys], y=[d.get(y) for y in ys],
                  name=label, marker_color=color)
    # ══ B11-3 (KRİTİK) ═══════════════════════════════════════════════
    # Plotly'nin varsayılanı rangemode="normal" — eksen VERİDEN hesaplanır
    # ve SIFIRI GARANTİ ETMEZ. Çubuk uzunluğu değeri temsil ettiği için
    # kırpılmış taban, farkları görsel olarak şişirir.
    # ÖLÇÜM (Pandey vd., CHI 2015): kırpılmış çubuk grafikte algılanan
    # fark 2.77 vs sıfır-tabanlı 1.45 → ~%91 ABARTI (r=0.37). Correll vd.
    # (CHI 2020): etki, kırpma AÇIKÇA işaretlense bile sürüyor.
    f.update_yaxes(rangemode="tozero")
    f.update_layout(barmode="group", title="Gelir • Net Kâr • FCF (yıllık)",
                    **_LAYOUT)
    return f


def fig_margins(m):
    f = go.Figure()
    for key, label, color in (("gm", "Brüt", WONG["gok"]),
                              ("om", "Faaliyet", WONG["turuncu"]),
                              ("nm", "Net", WONG["mor"])):
        d = m[key]
        ys = sorted(d.keys())
        f.add_scatter(x=[str(y) for y in ys], y=[d[y] * 100 for y in ys],
                      name=label, mode="lines+markers",
                      marker=dict(size=8),          # B11-9: nokta GÖRÜNÜR
                      line=dict(width=2, color=color))
    # B11-9: çizgi, noktalar ARASINI ölçülmüş gibi gösterir. 3-4 yıllık
    # seride bu, olmayan bir "trend" izlenimi yaratabilir. Nokta sayısı
    # azsa başlıkta AÇIKÇA yazılır; trend çizgisi ASLA eklenmez.
    _n = max((len(m.get(k) or {}) for k in ("gm", "om", "nm")), default=0)
    f.update_layout(
        title=("Marj Mimarisi (%)"
               + (f"  ·  yalnız {_n} yıl — çizgi TREND DEĞİL, noktaları "
                  f"birleştirir" if _n and _n < 5 else "")),
        yaxis_ticksuffix="%", **_LAYOUT)
    return f


def fig_sensitivity(m):
    s = m.get("sens")
    if not s:
        return None
    grows = [f"{g * 100:.0f}%" for g in s["grows"]]
    discs = [f"{d * 100:.0f}%" for d in s["discs"]]
    z = [[(v if v is not None else np.nan) for v in row] for row in s["z"]]
    text = [[("—" if v is None else f"{v:,.0f}") for v in row] for row in s["z"]]
    # ══ B11-1 (KRİTİK — kodun kendi taahhüdüyle çelişiyordu) ═════════
    # Dosyanın başında "erkeklerin ~%8'i kırmızı-yeşil ayırt edemez →
    # durum ASLA yalnız renkle kodlanmaz" yazıyor. Ama bu ısı haritası
    # tam da onu yapıyordu: RdYlGn, ColorBrewer'ın renk körü GÜVENLİ
    # OLMAYAN listesinde.
    # ÖLÇÜM (Viénot-Brettel-Mollon simülasyonu, uç noktalar):
    #   RdYlGn: normal 328 → dötanopi  50  (%85 çöküş — uçlar aynı)
    #   RdBu  : normal 206 → dötanopi 124  (korunuyor)
    # RdBu hem güvenli hem "sıcak=pahalı / soğuk=ucuz" sezgisini korur.
    # zmid: mevcut fiyat, renk skalasının NÖTR noktasına oturtulur —
    # yoksa "yeşil" bölge veriye göre kayar ve anlam değişir.
    _zf = [v for r in s["z"] for v in r if v is not None]
    _fiyat = _num(m.get("price"))          # B11-1: 'price' AŞAĞIDA tanımlı
    _zmid = float(_fiyat) if _fiyat else (
        (min(_zf) + max(_zf)) / 2 if _zf else None)
    f = go.Figure(go.Heatmap(
        z=z, x=grows, y=discs, text=text, texttemplate="%{text}",
        colorscale="RdBu", zmid=_zmid, showscale=False))
    price = m.get("price")
    sub = f" · Mevcut fiyat: {m['cur']}{price:,.2f}" if price else ""
    f.update_layout(title="DCF Duyarlılık — Adil Değer/Hisse "
                          f"(satır: iskonto, sütun: büyüme){sub}",
                    **_LAYOUT)
    return f


def fig_gauge(m):
    """B11-6: gösterge + belirsizlik bandı.

    NOT: Görselleştirme literatürü gösterge/hız-saati grafiklerini büyük
    ölçüde ELEŞTİRİR (Few: mermi grafiği tam da bunu değiştirmek için
    tasarlandı; dairesel alan tek bir sayı için çok yer harcar).
    Burada gösterge KORUNDU çünkü tanıdıklığı yüksek ve nitel bantları
    tek bakışta veriyor — ama sahte kesinlik ve renk körlüğü sorunları
    giderildi ve altına belirsizlik bandı yazıldı.
    """
    score = m["score"]
    if score is None:
        return None
    # ══ B11-6 ════════════════════════════════════════════════════════
    # İki sorun vardı:
    # (a) SAHTE KESİNLİK. "gauge+number" sezgisel bir skoru ondalıklı
    #     kesinlikle basıyordu. Witteman vd. (JMIR 2011, N=3.422):
    #     ondalıklı yerine TAM SAYI gösterilen tahminleri %7-10 daha çok
    #     kişi "yüksek güvenilir" buluyor ve daha iyi hatırlıyor —
    #     yani ondalık, hak edilmemiş bir kesinlik sinyali veriyor.
    # (b) RENK KÖRLÜĞÜ. Yeşil (#2E7D32) ile kırmızı (#C62828) bantların
    #     algısal farkı normal görüşte 294, dötanopide 52 (%82 çöküş):
    #     ikisi de zeytin rengine dönüyor (#6D6D35 vs #767618).
    # ÇÖZÜM: tam sayı + tek-ton (mavi→turuncu) sıralı bantlar + eşik
    # çizgisi. Bantlar artık AÇIKTAN KOYUYA sıralı, yani gri tonlamada
    # ve her renk körlüğü türünde okunuyor.
    _bant = m.get("skor_kararlilik") or {}
    if _bant.get("hata"):
        _bant = {}   # hesaplanamadıysa band gösterme
    f = go.Figure(go.Indicator(
        mode="gauge+number", value=round(float(score)),
        number={"suffix": " /100", "valueformat": ".0f"},
        title={"text": "Pahalılık Skoru<br>"
                       "<span style='font-size:0.75em'>SIRALAMA aracı — "
                       "eşikler sınandı ve KALDI</span>"},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color="#ECEFF1", thickness=0.22),
            # açıktan koyuya SIRALI — gri tonlamada da okunur
            steps=[dict(range=[0, 35], color="#D6E9F5"),
                   dict(range=[35, 55], color="#9CC7E0"),
                   dict(range=[55, 70], color="#E8B778"),
                   dict(range=[70, 85], color="#C97B2E"),
                   dict(range=[85, 100], color="#8A4B12")],
            threshold=dict(line=dict(color="#F0E442", width=3),
                           thickness=0.85, value=float(score)),
        )))
    # B11-6: belirsizlik bandı — nokta değil BANT çıpa olsun
    if _bant.get("bant_p5") is not None:
        f.add_annotation(
            x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
            font=dict(size=11, color="#BDBDBD"),
            text=(f"belirsizlik bandı {_bant['bant_p5']:.0f}–"
                  f"{_bant['bant_p95']:.0f} · hüküm kararlılığı "
                  f"%{_bant['hukum_kararliligi']*100:.0f}"))
    f.update_layout(template=_plotly_tema(), height=300,
                    margin=dict(l=30, r=30, t=60, b=10))
    return f


# ---------------------------------------------------------------------------
# 10) ARAYÜZ
# ---------------------------------------------------------------------------

def render_girdiler():
    """Tek Hisse girdileri — SAYFA İÇİ kontrol paneli. Eski kenar
    çubuğu buraya taşındı; mod seçimi artık üst navigasyonda."""
    kap = st.container(border=True)
    kap.markdown("#### ⚙️ Analiz Girdileri")
    sol, sag = kap.columns([1.05, 1])
    sb = sol
    # Başlangıç değeri value= yerine session_state ile verilir; böylece
    # köprünün (_git_tkr) rerun'da bu key'e yazması uyarı/hata üretmez.
    st.session_state.setdefault("tkr", "AAPL")
    ticker = sb.text_input("Ticker (ABD — ör. AAPL, NVDA)",
                           key="tkr",
                           help="Yalnız ABD borsaları (NYSE/NASDAQ): "
                                "AAPL, NVDA, MSFT, JPM…").strip().upper()
    run = sb.button("🔍 Analizi Çalıştır", type="primary",
                    width="stretch")
    edu = sb.toggle("📚 Eğitici mod — metrikler ne demek?", value=True,
                    help="Açıkken her metrikte sade açıklama, sekmelerde "
                         "okuma rehberi ve Claude raporunda terim izahı olur.")
    use_edgar = sb.checkbox(
        "🏛️ SEC EDGAR çapraz doğrulama", value=True,
        help="Kritik rakamları (gelir, kâr, nakit, özsermaye, hisse) devletin "
             "resmî 10-K verisiyle karşılaştırır — uyuşmazlıkta SEC kazanır. "
             "Kapatırsan analiz yalnız Yahoo ile, biraz daha hızlı çalışır.")
    peers_raw = sb.text_input(
        "👥 Rakipler (virgülle, opsiyonel)", value="",
        help="Ör: MSFT, GOOGL — girersen çarpanlar rakip medyanıyla "
             "kıyaslanır. Boş bırakırsan Claude raporu rakipleri webden "
             "kendisi bulur.")
    peers = tuple(p.strip().upper() for p in peers_raw.split(",")
                  if p.strip())[:6]
    maliyet = sb.number_input(
        "💼 Ortalama maliyetin (opsiyonel, $)", min_value=0.0, value=0.0,
        step=0.01, format="%.2f",
        help="Bu hisseyi elinde tutuyorsan giriş maliyetini yaz: çıkış "
             "planında güncel kâr/zarar ve dilim kazançları gösterilir. "
             "Uyarı: maliyet bilgi içindir, karar çıpası DEĞİLDİR.")
    ufuk = sb.selectbox(
        "⏱️ Planlanan tutma süresi", list(UFUKLAR.keys()), index=0,
        help="Analizi ufkuna göre çerçeveler: tipik dalgalanma bandı, ufka "
             "düşen bilançolar, zaman stopu tarihi ve kısa ufukta temel "
             "analizin sınırları. YÖN tahmini yapmaz — 'düşer mi?' "
             "sorusunun cevabını kimse bilemez; bilinebilir olan aralık, "
             "takvim ve karar kurallarıdır.")
    with sb.expander("🎙️ Kazanç Çağrısı Metni (opsiyonel)"):
        call_text = st.text_area(
            "Son çağrının/basın bülteninin metnini yapıştır",
            value="", height=140, key="call_txt",
            help="IR sayfasından transcript/özet yapıştırırsan dil analizi "
                 "(hedge vs güven kelimeleri) yapılır ve Claude raporuna "
                 "girer. Boş bırakırsan Claude webden arar.")

    sugg = st.session_state.get("sugg")
    tkey = st.session_state.get("tkr", "GENEL")
    with sag.expander("📐 DCF Varsayımları", expanded=False):
        # (a2) Damodaran ima edilen ERP'yi HER AY yayınlar; sabit tarihsel
        # ERP kullanmak eskir. Rf de 10Y Hazine ile birlikte oynar.
        # ── CANLI 10Y — kenar çubuğu varsayılanı buradan gelir ───────
        _canli = canli_10y()
        _rf_var = (_canli["yuzde"] if _canli
                   else float(RISK_FREE_RATE * 100))
        if _canli:
            st.success(f"📡 10Y canlı: **%{_canli['yuzde']:.2f}** "
                       f"({_canli['tarih']}) — otomatik çekildi")
        else:
            _bay = rf_erp_bayatlik()
            st.warning(f"📡 Canlı 10Y ÇEKİLEMEDİ — modül sabiti "
                       f"(%{RISK_FREE_RATE*100:.2f}, {RF_ERP_TARIH}"
                       + (f", {_bay} ay önce" if _bay else "")
                       + ") kullanılıyor. Değeri elle güncellemen "
                         "gerekebilir.")
        st.caption("⚠️ **Fed faizi ≠ bu alan.** Fed GECELİK politika "
                   "faizini belirler; DCF 10 YILLIK Hazine getirisini "
                   "kullanır. 10Y kararı önceden fiyatlar ve Fed "
                   "artırırken bile düşebilir — 'Fed açıkladı, elle "
                   "değiştireyim' yerine bu alanın canlı değerine bak.")
        cc1, cc2 = st.columns(2)
        cc1.number_input(
            "Rf %", 1.0, 8.0, _rf_var, 0.1,
            key="rf_in",
            help="Risksiz faiz — 10 yıllık ABD Hazine getirisi (Fed'in "
                 "gecelik faizi DEĞİL). Varsayılan CANLI çekilir; elle "
                 "değiştirirsen senin değerin kazanır.")
        cc2.number_input(
            "ERP %", 2.0, 8.0, float(EQUITY_RISK_PREMIUM * 100), 0.05,
            key="erp_in",
            help="İma edilen hisse risk primi. Damodaran her ay günceller: "
                 "pages.stern.nyu.edu/~adamodar — güncel ima edilen ERP, "
                 "sabit tarihsel ERP'den daha iyi pratiktir.")
        auto = st.checkbox(
            "🤖 Otomatik varsayım (şirkete göre)", value=True,
            help="Motor; 3Y gelir CAGR'ı, son yıl büyümeyi ve analist "
                 "beklentisini harmanlar; ölçek, kaldıraç ve kârlılığa göre "
                 "iskonto belirler. Kapat → öneri başlangıç değeri olarak "
                 "yüklenir, son sözü sen söylersin.")
        if sugg:
            st.caption(f"**Önerilen:** büyüme %{sugg['growth']*100:.1f} · "
                       f"iskonto %{sugg['discount']*100:.1f} · "
                       f"{sugg['years']} yıl projeksiyon")
            st.caption(f"Gerekçe → büyüme: {sugg['_why_growth']} · "
                       f"iskonto: {sugg['_why_discount']}")
        d = sugg or {"growth": 0.08, "discount": 0.10,
                     "terminal": 0.025, "years": 5}
        if auto:
            growth = d["growth"] * 100
            discount = d["discount"] * 100
            terminal = d["terminal"] * 100
            years = d["years"]
            if not sugg:
                st.caption("Öneri, ilk analizden sonra şirkete göre "
                           "hesaplanır (şimdilik jenerik taban: %8 / %10).")
        else:
            growth = st.slider("FCF büyümesi (yıllık, %)", -5.0, 40.0,
                               float(round(d["growth"] * 100, 1)), 0.5,
                               key=f"g_{tkey}", help=GLOSSARY["buyume_var"])
            discount = st.slider("İskonto oranı (%)", 6.0, 18.0,
                                 float(round(d["discount"] * 100, 1)), 0.5,
                                 key=f"d_{tkey}", help=GLOSSARY["iskonto_var"])
            terminal = st.slider("Terminal büyüme (%)", 1.0, 4.0,
                                 float(round(d["terminal"] * 100, 2)), 0.25,
                                 key=f"t_{tkey}", help=GLOSSARY["terminal_var"])
            years = st.slider("Projeksiyon yılı", 3, 10, int(d["years"]),
                              key=f"y_{tkey}", help=GLOSSARY["yil_var"])
        oto_g = st.checkbox(
            "Büyümeyi şirketin kendi geçmişinden hesapla", value=True,
            help="Sabit %8 yerine şirketin FCF/gelir CAGR'ından türetir "
                 "(temkinli: düşüğü alır; taban −%5, tavan kaliteye göre "
                 "%15/18/22). "
                 "Kapatırsan yukarıdaki slider değeri kullanılır.")
        sbc_adjust = st.checkbox(
            "SBC'yi FCF'ten düş (temkinli)", value=True,
            help="Hisse bazlı tazminat 'nakit dışı'dır ama hissedarı "
                 "dilüsyonla cezalandırır; düşmek disiplindir.")
        nd_adjust = st.checkbox(
            "Net borcu değerden düş (ÇİFTE CEZA — varsayılan kapalı)",
            value=False,
            help="CFO−CapEx zaten faiz ödenmiş (kaldıraçlı) bir akıştır; "
                 "net borcu AYRICA düşmek borçlu şirkete çifte ceza verir. "
                 "5 hisselik testte bu, adil değeri o kadar düşürdü ki "
                 "program hiçbirine 'al' diyemedi. Açarsan çok temkinli "
                 "olursun — kaliteli şirketler sürekli 'pahalı' görünür.")

    with sag.expander("🧠 Claude Ayarları", expanded=False):
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("API anahtarı", type="password",
                                value="", placeholder=("ortamdan alındı ✓"
                                                       if env_key else "sk-ant-..."))
        api_key = api_key or env_key
        models = st.session_state.get("model_list", DEFAULT_MODELS)
        model = st.selectbox("Model", models, index=0)
        if st.button("Hesabımdaki modelleri listele",
                     width="stretch"):
            if not api_key:
                st.warning("Önce API anahtarı gir.")
            else:
                try:
                    st.session_state["model_list"] = list_account_models(api_key)
                    st.rerun()
                except Exception as e:
                    st.error(f"Liste alınamadı: {e}")
        use_web = st.checkbox("Claude web araması yapsın (güncel SEC/haber)",
                              value=True,
                              help="API tarafında ek arama ücreti doğurur; "
                                   "hesapta kapalıysa otomatik geri düşülür.")

    with sag.expander("ℹ️ Veri kaynağı & sınırlar"):
        st.markdown(
            "- Kaynak: **Yahoo Finance (yfinance)** — hızlı tarama katmanı; "
            "SEC dipnot derinliği **değildir**.\n"
            "- Fiyatlar gecikmeli olabilir; bazı ticker'larda satırlar eksik "
            "gelebilir (motor eksikleri raporlar).\n"
            "- Skor **sezgiseldir**; Claude katmanı skoru denetlemekle "
            "görevlidir.\n"
            "- Bu araç **yatırım tavsiyesi değildir**.")

    with sag.expander("🛡️ Stop & Pozisyon (kanıta dayalı)", expanded=False):
        stop_acik = st.toggle(
            "Stop & Destek modülü", value=True,
            help="ATR-22 Chandelier trailing + destek-bandı-altı giriş "
                 "stopu + risk-bazlı pozisyon boyutu. Kaminski-Lo uyarısı: "
                 "stop momentum rejiminde değer katar, ortalamaya dönüşte "
                 "getiriyi DÜŞÜREBİLİR — bu yüzden kapatılabilir.")
        st.caption("✅ Aşağıdaki varsayılanlar 34 hisse × 15 yıl "
                   "walk-forward testiyle DOĞRULANDI (out-of-sample "
                   "beklenti +0.28R/işlem, SQN 1.77).")
        atr_carpan = st.slider(
            "Trailing çarpan (×ATR22)", 2.0, 4.0, 3.0, 0.5,
            help="Chandelier varsayılanı 3.0 (LeBeau). Yüksek çarpan "
                 "whipsaw'ı azaltır ama potansiyel zararı büyütür.")
        giris_carpan = st.slider(
            "Giriş stopu çarpanı (×ATR22)", 2.0, 4.0, 3.0, 0.5,
            help="DOĞRULANMIŞ: testte 3.0-3.5× dar çarpanları (2.0-2.5) "
                 "belirgin biçimde yendi. Dar stop erken çıkarır.")
        r_hedef = st.slider(
            "Kâr alma hedefi (×R)", 1.5, 4.0, 3.0, 0.5,
            help="DOĞRULANMIŞ: 3R, direnç-bazlı hedefi açık farkla yendi. "
                 "İsabet düşer (~%15) ama kazananlar büyük olur.")
        trend_filtre = st.toggle(
            "MA200 trend filtresi", value=False,
            help="Yalnız fiyat 200 günlük ortalamanın ÜSTÜNDEYKEN alım "
                 "sinyali üret. Testte beklentiyi hafif düşürdü ama işlem "
                 "sayısını ~%37 azaltıyor — daha az sarsıntı, daha az "
                 "fırsat. Ayı piyasasından korur.")
        portfoy = st.number_input(
            "Portföy büyüklüğü (opsiyonel, $)", min_value=0.0, value=0.0,
            step=1000.0, format="%.0f",
            help="Girersen pozisyon boyutu somut hesaplanır; boşsa "
                 "'1.000$ risk başına' örneği gösterilir.")
        risk_pct = st.slider(
            "İşlem başına risk (%)", 0.5, 3.0, 1.0, 0.25,
            help="%1-2 kuralı SAVUNMACIDIR (Tharp); optimal risk "
                 "stratejiye bağlıdır (Kelly). Hayatta kalma aracı, "
                 "getiri garantisi değil.")
    a = {"growth": growth / 100, "discount": discount / 100,
         "terminal": terminal / 100, "years": int(years),
         "sbc_adjust": sbc_adjust, "net_debt_adjust": nd_adjust,
         "oto_buyume": oto_g,
         "ufuk": ufuk, "stop_acik": stop_acik, "atr_carpan": atr_carpan,
         "giris_carpan": giris_carpan, "r_hedef": r_hedef,
         "trend_filtre": trend_filtre,
         "portfoy": portfoy, "risk_pct": risk_pct}
    with sag.expander("📓 Kayıt Defteri (kalibrasyon)"):
        gk = gunluk_oku()
        if gk:
            st.caption(f"{len(gk)} analiz kaydı · `{GUNLUK_DOSYA}` "
                       f"dosyasında (uygulama klasörü). Aynı hisseyi tekrar "
                       f"çalıştırdığında önceki hükümle kıyas yapılır.")
            st.dataframe(pd.DataFrame([
                {"Tarih": k.get("tarih", "")[:10],
                 "Ticker": k.get("ticker"),
                 "Fiyat": k.get("fiyat"),
                 "Skor": k.get("skor"),
                 "Hüküm": (k.get("hukum") or "")[:28]}
                for k in gk[-8:]][::-1]),
                hide_index=True, width="stretch")
            bekleyen = [k for k in gk if k.get("tip") == "karar"
                        and not k.get("cozuldu")]
            if bekleyen and st.button("🔄 Ufku dolan kararları çöz"):
                def _h(t):
                    try:
                        return fetch_bundle(t, False).get("hist")
                    except Exception:
                        return None
                n_ = kayit_coz(gk, _h)
                if n_:
                    try:
                        with open(GUNLUK_DOSYA, "w", encoding="utf-8") as f:
                            json.dump(gk[-200:], f, ensure_ascii=False,
                                      indent=1)
                        st.success(f"{n_} karar çözüldü.")
                    except Exception:
                        st.warning("Sonuçlar hesaplandı ama dosyaya "
                                   "yazılamadı.")
                else:
                    st.info("Ufku dolmuş çözülebilir karar yok (veya fiyat "
                            "serisi alınamadı).")
            kal = kalibrasyon(gk)
            if kal:
                st.markdown("**🎯 Kalibrasyon**")
                kk1, kk2 = st.columns(2)
                kk1.metric("Brier skoru", f"{kal['brier']:.3f}",
                           help="0 mükemmel · 0.25 yazı-tura. Düşük iyidir.")
                kk2.metric("Çözülmüş karar", kal["adet"])
                st.dataframe(pd.DataFrame(kal["bantlar"]).rename(columns={
                    "bant": "Olasılık bandı", "adet": "Adet",
                    "gerceklesen": "Gerçekleşen %",
                    "beklenen": "Beklenen %"}),
                    hide_index=True, width="stretch")
                (st.info if kal["yeterli_mi"] else st.warning)(kal["not"])
            son_pre = [k for k in gk if k.get("premortem")]
            if son_pre:
                with st.expander("🧨 Son premortem kayıtların"):
                    for k in son_pre[-3:][::-1]:
                        st.caption(f"**{k.get('ticker')}** · "
                                   f"{k.get('tarih', '')[:10]} · "
                                   f"olasılık %{k.get('olasilik')}")
                        st.text(k["premortem"][:400])
        else:
            st.caption("Henüz kayıt yok — her 'Analizi Çalıştır' bir kayıt "
                       "düşer; zamanla hükümlerin isabeti izlenir.")
    return (ticker, run, a, api_key, model, use_web, edu, use_edgar,
            peers, call_text, maliyet)


ANALIZ_AKISI = ("1·Veri sağlığı", "2·Kalite & adli", "3·Değerleme",
                "4·Karar & plan", "5·Şirketi anla (5 Soru)",
                "6·Kayda geç")


# ═══════════════════════════════════════════════════════════════════════
# SADE ÖZET KATMANI — "bir bakışta anla" (araştırma: sade-dil raporu)
# İlkeler: progressive disclosure (Nielsen 1995; detay sekmelerde kalır),
# doğal frekans + referans sınıfı (Gigerenzer-Hoffrage 1995; Gigerenzer
# 2005 '%30 yağmur' çalışması), sayısal aralıkla belirsizlik (van der
# Bles 2020 PNAS: aralık göstermek kaynağa güveni zedelemez), her etikete
# belirsizlik eşlik eder (Morningstar Uncertainty Rating deseni), kısa ve
# güçlü uyarı (Mercer 2010: gömülü standart uyarı etkisiz). Cümleler
# DETERMİNİSTİK şablondur — LLM yok, yalnız zaten hesaplanmış alanlar.
# ═══════════════════════════════════════════════════════════════════════

def skor_kelimesi(kalite):
    """Kalite 0-100 → 5 kademeli kelime (tam etiketleme: Alwin-Krosnick
    1991; kademe sayısı: Cowan 2001 ~4 parça / Preston-Colman 2000)."""
    if kalite is None:
        return {"kelime": "kalitesi ölçülemeyen", "ikon": "⚪"}
    if kalite >= 80:
        return {"kelime": "çok güçlü", "ikon": "🟢"}
    if kalite >= 60:
        return {"kelime": "güçlü", "ikon": "🟢"}
    if kalite >= 40:
        return {"kelime": "orta/karışık görünen", "ikon": "🟡"}
    if kalite >= 20:
        return {"kelime": "zayıf görünen", "ikon": "🟠"}
    return {"kelime": "çok zayıf/riskli görünen", "ikon": "🔴"}


def bant_kelimesi(score):
    """Pahalılık 0-100 → KONUM kelimesi (verdict_info bantlarıyla aynı
    eşikler; dil 'hüküm' değil 'bant' — koddaki 30 Tem 2026 doktrini)."""
    if score is None:
        return "fiyat konumu ölçülemiyor"
    if score < 35:
        return "fiyatı ucuz bantta görünüyor"
    if score < 55:
        return "fiyatı makul bantta"
    if score < 70:
        return "fiyatı yüksek bantta görünüyor"
    return "fiyatı çok yüksek bantta görünüyor"


def sade_ozet(m):
    """TEK HİSSE sade özeti — 5 bileşen: (1) tek cümlelik cevap,
    (2) ne demek / ne demek DEĞİL, (3) en fazla 3 işaret (Cowan 2001),
    (4) tek cümlelik plan, (5) güven etiketi + dürüstlük dipnotu.
    YALNIZ m'de zaten hesaplanmış alanlardan beslenir; None dönebilir."""
    if not m:
        return None
    kalite, score = m.get("quality"), m.get("score")
    price, cur = m.get("price"), m.get("cur") or "$"
    sk = skor_kelimesi(kalite)

    # (1) tek cümle
    cumle = f"{sk['ikon']} {sk['kelime'].capitalize()} bir şirket; " \
            f"{bant_kelimesi(score)}."

    # (2) ne demek / ne demek değil — (kalite, skor) çeyreğine göre
    hi_q = kalite is not None and kalite >= 65
    lo_q = kalite is not None and kalite < 40
    if score is None:
        nd = ("Değerleme skoru bu hissede hesaplanamadı (veri eksik).",
              "Skor yok diye 'sorun yok' demek DEĞİL — kör noktadayız, "
              "temkin payı büyütülür.")
    elif score < 35 and lo_q:
        nd = ("Fiyat düşük bantta AMA şirketin sağlığı zayıf — ucuzluğun "
              "bir sebebi olabilir (değer tuzağı riski).",
              "'Ucuz = fırsat' demek DEĞİL. Ucuzluk çoğu zaman artan "
              "riskin fiyata yansımasıdır; önce neden ucuz sorusu.")
    elif score < 35:
        nd = ("Fiyat, aracın değer bandının altında; kalite tarafı da "
              "kötü görünmüyor — nadir bir kombinasyon.",
              "'Kesin yükselir' demek DEĞİL. Piyasa bir şey görüp iskonto "
              "veriyor olabilir; en kötü senaryo yine de test edilir.")
    elif score < 55:
        nd = ("Fiyat adil değer bandının içinde; ucuzluk avantajı yok, "
              "aşırılık da yok.",
              "Bu bir alım çağrısı DEĞİL — bu bantta kenar (edge) fiyattan "
              "değil, işin kalitesinden gelmek zorunda.")
    elif score < 70:
        nd = ("Bugünkü fiyat, gelecekten çok şey bekliyor; güvenlik payı "
              "dar.",
              "'Yarın düşer' demek DEĞİL. Pahalı bant bir zamanlama "
              "kehaneti değildir; yalnızca hata payının azaldığını söyler.")
    else:
        nd = ("Fiyat, aracın ölçtüğü değerin çok üzerinde — bu seviyeden "
              "alım güvenlik payı bırakmıyor.",
              "'Açığa sat' sinyali DEĞİL. Çok pahalı bant yön tahmini "
              "değildir; sadece bu fiyattan ALIM tartışılmaz der.")

    # (3) işaretler — öncelik: kırmızılar önce, sonra sarı/yeşil; en çok 3
    kirmizi, sari, yesil = [], [], []
    pf = m.get("piotroski") or {}
    if pf.get("max", 0) >= 6 and pf.get("score") is not None:
        if pf["score"] <= 3:
            kirmizi.append(f"🔴 Finansal sağlık zayıf (F-skor "
                           f"{pf['score']}/{pf['max']})")
        elif pf["score"] >= 7:
            yesil.append(f"🟢 Finansal sağlık güçlü (F-skor "
                         f"{pf['score']}/{pf['max']})")
    z = m.get("altman")
    if z is not None:
        _ab = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                               ALTMAN_BOLGE["uretim"])
        if z < _ab["tehlike"]:
            kirmizi.append(f"🔴 İflas-mesafesi bandı riskli (Z {z:.2f})")
        elif z >= _ab["guvenli"]:
            if m.get("altman_x4_uyarisi"):
                sari.append("🟡 Z 'güvenli' diyor ama kârlılık negatif — "
                            "Z'nin X4'ü piyasa değeriyle şişmiş olabilir")
            else:
                yesil.append(f"🟢 İflas-mesafesi bandı güvenli (Z {z:.2f})")
    fcf = m.get("fcf_base")
    if fcf is not None:
        if fcf <= 0:
            kirmizi.append("🔴 Serbest nakit akışı negatif — fiyat vaade "
                           "yaslanıyor")
        else:
            yesil.append("🟢 Serbest nakit akışı pozitif")
    try:
        _h = m.get("_hist")
        _c = _h["Close"].dropna() if (_h is not None and "Close" in _h) \
            else None
        if _c is not None and len(_c) >= 200 and price:
            if price < float(_c.rolling(200).mean().iloc[-1]):
                sari.append("🟡 Fiyat 200 günlük ortalamanın altında "
                            "(trend bağlamı — sinyal değil)")
    except Exception:
        pass
    isaretler = (kirmizi + sari + yesil)[:3] or \
        ["⚪ Belirgin bir işaret yok — detay sekmelere bak"]

    # (4) plan cümlesi — kademeli_plan alanlarından
    kd = m.get("kademe") or {}
    d1 = ((kd.get("dilimler") or [{}])[0]).get("seviye")
    esik = kd.get("izleme_esigi")
    hedef = d1 or esik
    if hedef and price:
        fark = (1 - hedef / price) * 100
        yer = f"~{cur}{hedef:,.2f} (bugünkü fiyatın %{fark:.0f} altı)" \
            if fark >= 0.5 else f"~{cur}{hedef:,.2f} (bugünkü fiyata yakın)"
        if kd.get("mod") == "KADEMELİ_GİRİŞ":
            plan = f"Plan kademeli giriş öneriyor — ilk dilim bölgesi {yer}."
        elif kd.get("mod") == "SINIRDA_BEKLE":
            plan = (f"İlk dilim bölgesi {yer} — oraya kadar izleme; "
                    f"acele yok.")
        else:
            plan = (f"Bu banttan alım planı YOK; izleme eşiği {yer} — "
                    f"eşik gelmeden alım tartışılmaz.")
    else:
        plan = "Plan seviyesi üretilemedi (veri eksik) — detay sekmesine bak."

    # (5) güven etiketi — veri kapsamından (basit ≠ aşırı güvenli)
    kapsam = sum([
        kalite is not None, score is not None,
        m.get("fair") is not None, pf.get("max", 0) >= 6,
        z is not None])
    guven = ("güven: yüksek" if kapsam >= 4 else
             "güven: orta" if kapsam >= 3 else "güven: düşük")

    return {"cumle": cumle, "ne_demek": nd[0], "ne_demek_degil": nd[1],
            "isaretler": isaretler, "plan": plan, "guven": guven,
            "dipnot": ("Bu bir tahmin değil, olasılık çerçevesidir. "
                       "Yatırım tavsiyesi değildir.")}


def render_sade_ozet(m):
    """Sade Özet kartı — karar ekranının EN ÜSTÜNDE (özet önce, detay
    sekmelerde: progressive disclosure)."""
    so = sade_ozet(m)
    if not so:
        return
    with st.container(border=True):
        st.markdown(f"### 🗣️ Sade Özet &nbsp; "
                    f"<small>({so['guven']})</small>",
                    unsafe_allow_html=True)
        st.markdown(f"**{so['cumle']}**")
        st.markdown(f"**Ne demek:** {so['ne_demek']}")
        st.markdown(f"**Ne demek DEĞİL:** {so['ne_demek_degil']}")
        for i in so["isaretler"]:
            st.markdown("- " + i)
        st.markdown(f"**Şimdi ne yapılır?** {so['plan']}")
        st.caption("⚖️ " + so["dipnot"])


def taban_orani_sade(tb):
    """Taban oranlarını DOĞAL FREKANS + referans sınıfı + sayısal aralık
    cümlesine çevirir (Gigerenzer-Hoffrage 1995; Gigerenzer 2005;
    van der Bles 2020). tb = taban_oranlari(...) çıktısı."""
    try:
        if not tb or tb.get("yetersiz"):
            return None
        n_et = int(tb.get("etkin_gozlem") or 0)
        poz = tb.get("pozitif_oran")
        med = tb.get("medyan")
        p10 = tb.get("p10")
        if None in (poz, med, p10) or n_et <= 0:
            return None
        k = int(round(poz * n_et))
        lo, hi = _wilson(k, n_et)
        yuz = int(round(poz * 100))
        c = (f"🗣️ **Sade okuma:** Geçmişte **buna benzer** (bugünkü "
             f"çekilmeye yakın) 100 dönemin yaklaşık **{yuz} tanesi** "
             f"3 ay sonra artıda bitti. Ortadaki durumda getiri "
             f"%{med*100:+.0f} civarıydı; **en kötü 10 dönemde kayıp "
             f"%{abs(p10)*100:.0f}'i aştı.**")
        if lo is not None:
            c += (f" Bu oran kabaca **%{lo*100:.0f}–%{hi*100:.0f}** "
                  f"arasında olabilir — yaklaşık {n_et} bağımsız benzer "
                  f"döneme dayanıyor" +
                  ("; az sayıda, dikkatli yorumla." if n_et < 30 else "."))
        return c
    except Exception:
        return None


def piyasa_hava_ozeti(pk):
    """Piyasa Bağlamı için 3 satırlık 'bugünkü hava durumu' özeti.
    TEK metafora sadık kalınır (Reijnierse 2025: metafor değiştirmek
    işlemeyi bozar) ve her satır gerçek ölçüme bağlanır."""
    try:
        rj = (pk or {}).get("rejim") or {}
        vade = (pk or {}).get("vade") or {}
        kredi = (pk or {}).get("kredi") or {}
        pz = (pk or {}).get("pozisyon") or {}
        ust = rj.get("ma200_ustu")
        vy = rj.get("vix_yuzdelik")
        back = vade.get("durum") == "backwardation"
        hook = bool(vade.get("hook"))
        kz = kredi.get("z")
        stres = sum([ust is False, bool(back and not hook),
                     kz is not None and kz >= 1,
                     vy is not None and vy >= 0.90])
        if stres == 0 and ust:
            hava, ikon = "Sakin", "🌤️"
            r1 = ("Endeks 200 günlük ortalamasının üstünde; korku ve "
                  "kredi göstergeleri olağan bantta.")
        elif stres >= 2:
            hava, ikon = "Fırtınalı", "⛈️"
            r1 = ("Birden fazla stres göstergesi aynı anda yanık — "
                  "geçmişte bu birleşim geniş dalgalanma dönemlerine "
                  "denk geldi.")
        else:
            hava, ikon = "Karışık", "🌥️"
            r1 = ("Göstergeler karışık: ne tam sakin ne belirgin stres.")
        carpan = pz.get("carpan")
        if hook:
            r2 = ("Tempo: stres göstergesi tepe yapıp GERİ DÖNMÜŞ — "
                  "kademeli alım hızlandırılabilir (kesinlik değil).")
        elif back:
            r2 = ("Tempo: akut stres aktif — yeni dilimi beklet, "
                  "dönüş işaretini bekle.")
        elif carpan is not None:
            r2 = (f"Tempo: normal pozisyonun ~{carpan:.2f} katı makul; "
                  + ("alımları normal hızda yay."
                     if 0.95 <= carpan <= 1.05 else
                     "biraz daha temkinli boyutlan."
                     if carpan < 0.95 else
                     "dağılım hafif lehte — yine de kademeli."))
        else:
            r2 = "Tempo: veri eksik — varsayılan kademeli disiplin."
        r3 = ("Bu 'hava durumu' HİSSEYE göre değişir: piyasayla bağı "
              "güçlü hisselerde sözü çok geçer; yeni/kendine özgü "
              "hisselerde az. Aşağıya sembol girersen R² ile ölçülür.")
        return {"baslik": f"{ikon} Bugünkü piyasa havası: **{hava}**",
                "satirlar": [r1, r2, r3],
                "not": ("Hava durumu bir YÖN tahmini değildir; yalnızca "
                        "boyut/tempo içindir ('timing değil, sizing').")}
    except Exception:
        return None


def render_karar_ekrani(m, a):
    """
    TEK KARAR EKRANI (araştırma A2): tüm modüllerin özeti 7 öğede.
    Bilişsel yük kanıtı (Cowan 2001: ~4 parça çekirdek kapasite) gereği
    aynı anda gösterilen bağımsız öğe sayısı sınırlı tutulur; detay
    sekmelere KATLANIR (silinmez).
    """
    st.caption("🧭 Analiz akışı: " + "  ›  ".join(ANALIZ_AKISI))

    # 0) SADE ÖZET — bir bakışta (sade-dil katmanı; detay sekmelerde)
    render_sade_ozet(m)

    # 1) Son ziyaretten beri değişenler
    fark = m.get("_fark")
    if fark:
        with st.container(border=True):
            st.markdown("**🔔 Son ziyaretten beri "
                        f"({fark.get('onceki_tarih', '—')})**")
            for tur, txt in (fark.get("degisimler") or [])[:6]:
                (st.warning if tur == "uyarı" else st.info)(txt)

    # 2) Hüküm + iki eksen
    render_verdict(m)

    # 3) Fiyat vs değer çıpaları
    st.markdown("##### 🎯 Fiyat ve Değer Çıpaları")
    mc = m.get("mc") or {}
    cip = [("Fiyat", m.get("price")), ("DCF baz", m.get("fair")),
           ("MC P25", mc.get("p25")), ("MC P50", mc.get("p50")),
           ("MC P75", mc.get("p75"))]
    c_ = st.columns(len(cip))
    for i, (ad, v) in enumerate(cip):
        c_[i].metric(ad, f"{m['cur']}{v:,.2f}" if v else "—",
                     (f"%{(v/m['price']-1)*100:+.0f}"
                      if (v and m.get("price") and ad != "Fiyat") else None),
                     delta_color="off")

    # 4) Plan tetikleri + ufuk
    _ck = ((m.get("kademe") or {}).get("cipa_kalitesi") or {})
    if _ck:
        (st.success if _ck.get("yapisal_dilim") else st.warning)(
            ("⚓ " if _ck.get("yapisal_dilim") else "⚠️ ") + _ck["not"])

    st.markdown("##### 🪜 Plan Tetikleri")
    kd, sp, uf = (m.get("kademe") or {}), (m.get("stop") or {}), (
        m.get("ufuk") or {})
    p1, p2, p3, p4 = st.columns(4)
    _d1i = ((kd.get("dilimler") or [{}])[0])
    d1 = _d1i.get("seviye")
    p1.metric("1. Alım Dilimi", f"{m['cur']}{d1:,.2f}" if d1 else
              (f"{m['cur']}{kd.get('izleme_esigi'):,.2f}"
               if kd.get("izleme_esigi") else "—"),
              help=(kd.get("mesaj") or "")[:180] or None)
    gs = (sp.get("giris_stop") or {}).get("seviye")
    p2.metric("Giriş Stopu", f"{m['cur']}{gs:,.2f}" if gs else "—")
    tr = (sp.get("trailing") or {}).get("seviye")
    p3.metric("Trailing Stop", f"{m['cur']}{tr:,.2f}" if tr else "—")
    p4.metric("Zaman Stopu", uf.get("zaman_stopu_tarihi", "—"),
              help=f"Ufuk: {uf.get('ufuk', '—')}")
    _by, _ba = uf.get("bant_yukari_yuzde"), uf.get("bant_asagi_yuzde")
    band = uf.get("tipik_bant_yuzde")
    if _by is not None and _ba is not None:
        st.caption(f"Tipik {uf.get('ufuk')} dalgalanma: yukarı "
                   f"**+%{_by*100:.0f}** / aşağı **−%{_ba*100:.0f}** "
                   f"(lognormal — simetrik değil). Bu bandın içi NORMAL "
                   f"gürültüdür, tez kırılması değildir.")
    elif band:
        st.caption(f"Tipik {uf.get('ufuk')} dalgalanma: ±%{band*100:.0f} "
                   f"· bu bandın içi NORMAL gürültüdür, tez kırılması "
                   f"değildir.")

    # 4b) Rejim & zamanlama (rapor G4/G5/G10) — skora girmez, tempo bilgisi
    _mo = m.get("momentum") or {}
    _pz = m.get("pead") or {}
    _parca = []
    if _mo.get("getiri_12_1") is not None:
        _parca.append(f"12-1 momentum %{_mo['getiri_12_1']*100:+.0f}")
    if _pz.get("son_surpriz_pct") is not None:
        _parca.append(f"son kazanç sürprizi %{_pz['son_surpriz_pct']:+.1f}"
                      + (" · pencerede"
                         if _pz.get("suruklenme_penceresinde") else ""))
    if m.get("trend_ok") is not None:
        _parca.append("MA200 " + ("üstü ✔" if m["trend_ok"] else "ALTI ✖"))
    if _parca:
        st.caption("🧭 Rejim & zamanlama (skora girmez): "
                   + " · ".join(_parca))
    if m.get("rejim_uyari"):
        st.warning("🌪️ Rejim uyarısı: " + "; ".join(m["rejim_uyari"])
                   + " — plan SEVİYELERİ değişmez, TEMPO düşürülür "
                     "(JT 1993 · Bernard-Thomas 1989).")

    # 5) Kritik uyarılar
    uyarilar = []
    if (m.get("tazelik") or {}).get("uyari"):
        uyarilar.append(("uyarı", m["tazelik"]["uyari"]))
    kb = (m.get("olaylar_8k") or {}).get("kirmizi_bayrak") or []
    for o in kb[:2]:
        its = ", ".join(f"{i.get('kod', '?')} {i.get('aciklama', '')}"
                        for i in (o.get("itemler") or []))
        uyarilar.append(("uyarı", f"SEC 8-K KIRMIZI BAYRAK "
                                  f"({o.get('tarih', '—')}): {its}"))
    if m.get("adr_notu"):
        uyarilar.append(("bilgi", "ADR/20-F: içeriden işlem ve çeyreklik "
                                  "SEC akışı bu hissede yoktur."))
    im = m.get("implied_move")
    if im and im.get("durum") == "OK":
        uyarilar.append(("bilgi", f"Kazanç ima edilen hareket: "
                                  f"±%{im.get('beklenen_hareket_pct', 0)*100:.1f} "
                                  f"({im.get('vade', '—')}) — beklenen MUTLAK "
                                  f"hareket (~%50 kapsama), 1σ DEĞİL; gerçek "
                                  f"1σ ±%{(im.get('bir_sigma_pct') or 0)*100:.1f} "
                                  f"ve daha geniştir."))
    for e in ((m.get("catalysts") or {}).get("gelecek") or [])[:1]:
        _gk = e.get("gun_kaldi")
        uyarilar.append(("bilgi", f"Yaklaşan: {e.get('olay', '—')} — "
                                  f"{e.get('tarih', '—')}"
                                  + (f" ({_gk} gün)" if _gk is not None
                                     else "")))
    if uyarilar:
        st.markdown("##### ⚠️ Kritik Uyarılar")
        for tur, txt in uyarilar[:5]:
            (st.warning if tur == "uyarı" else st.info)(txt)

    # 6) Kalite/adli tek satır
    st.markdown("##### 🧪 Kalite & Adli — Tek Bakış")
    k1, k2, k3, k4 = st.columns(4)
    pf = m.get("piotroski")
    k1.metric("Piotroski",
              (f"{pf.get('score')}/{pf.get('max', 9)}"
               if isinstance(pf, dict) and pf.get("score") is not None
               else "—"))
    # NOT: m["altman"] düz SAYIdır (sözlük değil) — bölge etiketi
    # kodun kendi eşikleriyle üretilir: <1.81 tehlike, <2.99 gri, üstü güvenli.
    al = _num(m.get("altman"))
    if al is None:
        k2.metric("Altman Z", "—",
                  "finansal sektörde n/a" if m.get("is_financial") else None,
                  delta_color="off")
    else:
        _esik = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                                 ALTMAN_BOLGE["uretim"])
        bolge = ("tehlike" if al < _esik["tehlike"] else
                 "gri bölge" if al < _esik["guvenli"] else "güvenli")
        if m.get("altman_model") == "hizmet":
            bolge += " (Z'')"
        k2.metric("Altman Z", f"{al:.1f}", bolge, delta_color="off")
    bn = m.get("beneish")
    if isinstance(bn, dict) and _num(bn.get("skor")) is not None:
        k3.metric("Beneish M", f"{_num(bn['skor']):.2f}",
                  bn.get("bolge"), delta_color="off")
    else:
        k3.metric("Beneish M", "—",
                  "finansal sektörde n/a" if m.get("is_financial") else None,
                  delta_color="off")
    fl = m.get("flags") or []
    k4.metric("Kırmızı Bayrak", len(fl))

    # 7) 5 Soru karnesi
    nit = st.session_state.get("claude_nitel")
    if nit:
        import re as _re
        mm = _re.search(r"Anlaşılırlık\s*Notu[^0-9]{0,12}(\d)\s*/\s*5",
                        nit)
        if mm:
            n_ = int(mm.group(1))
            (st.success if n_ >= 4 else st.warning)(
                f"🧩 Şirket anlaşılırlığı: **{n_}/5**"
                + ("" if n_ >= 4 else " — anlamadığın şirketi almamak da "
                                      "bir karardır."))
    else:
        st.caption("🧩 'Şirketi anla' turu henüz çalıştırılmadı — Claude "
                   "sekmesindeki **5 Soru** butonu.")
    st.caption("Detaylar aşağıdaki sekmelerde: değerleme motorları, "
               "finansallar, risk radarı, katalizör & haber, Claude raporu.")
    st.divider()


def render_verdict(m):
    label, emoji, kind, msg = m["verdict"]
    if not sahne_hukum(m):
        verdict_banner(label, emoji, kind, msg)
        _g1, _g2, _gb = st.columns([1, 1, 2.6])
        with _g1:
            if m.get("quality") is not None:
                gauge(m["quality"], "Kalite",
                      "ok" if m["quality"] >= 60 else
                      "warn" if m["quality"] >= 40 else "bad")
        with _g2:
            if m.get("score") is not None:
                gauge(m["score"], "Pahalılık",
                      "ok" if m["score"] < 35 else
                      "warn" if m["score"] < 55 else "bad")
        with _gb:
            st.empty()
    q = m.get("quality")
    sc = m.get("score")
    st.caption(f"İki eksenli hüküm → **Şirket kalitesi:** "
               f"{f'{q:.0f}/100' if q is not None else '—'} · "
               f"**Fiyat pahalılığı:** "
               f"{f'{sc:.0f}/100' if sc is not None else '—'}. Kalite 'iyi "
               f"şirket mi?', skor 'iyi fiyat mı?' sorusudur — ikisi ayrı "
               f"sorulardır ve hüküm daima fiyat ekseninden verilir.")

    # ── UFUK UYARISI ────────────────────────────────────────────────
    # Değerleme bileşikleri 1 yıl ve ÜZERİ ufukta çalışır; 3-4 ayda
    # beklenen kenarları yoktur. Frankel-Lee'nin V/P oranı 1. yılda
    # ~%3, 3. yılda ~%31 kümülatif makas veriyor — yani sinyal YILLAR
    # ölçeğinde birikiyor. Bartram-Grinblatt: değer yakınsaması 34 ayda
    # sönüyor. Bu, aracın kusuru değil değerlemenin doğası.
    st.caption(
        "⏳ **Ufuk:** Bu skor bir DEĞERLEME merceğidir ve değerleme "
        "sinyalleri 1 yıl ve üzeri ufuklarda çalışır (Frankel-Lee: 1. "
        "yıl ~%3, 3. yıl ~%31 kümülatif makas). 3-4 aylık bir tutma "
        "süresinde bu skorun beklenen kenarı YOKTUR — o ufukta fiyat "
        "hareketini momentum ve firmaya özgü haberler belirler. Skoru "
        "'ne zaman alayım' için değil, 'bu fiyatı ödemeye değer mi' "
        "için kullan.")

    # ── KALİBRASYON SONUCU (gerçek test, 30 Tem 2026) ───────────────
    _k = KALIBRASYON_SONUCU
    st.error(
        f"📏 **EŞİKLER SINANDI VE KALDI ({_k['gecen_olcut']} ölçüt).** "
        f"{_k['ozet']}\n\n"
        f"**{_k['sonuc']}**")
    with st.expander("📊 Kalibrasyon testinin ayrıntısı"):
        st.markdown(
            f"- **Yöntem:** {_k['hisse']} hisse, {_k['yil']}, "
            f"{_k['gozlem']:,} gözlem. SEC EDGAR'ın **noktasal-zaman** "
            f"(as-filed) verisi — yani o gün gerçekten bilinebilir olan "
            f"rakamlar. Yahoo'nun sonradan düzeltilmiş verisi DEĞİL.\n"
            f"- **Ölçütler veriye bakılmadan ÖNCE ilan edildi**, sonradan "
            f"değiştirilmedi.\n"
            f"- Monotonluk: Spearman **{_k['spearman']:+.2f}** "
            f"(gereken ≤ −0.60) ✖\n"
            f"- Makas: **%{_k['makas']*100:+.1f}**, t=**{_k['t_ist']:.2f}** "
            f"(gereken t>2.0) ✖\n"
            f"- En ucuz %20: **%{_k['ucuz_ceyrek']*100:.1f}** · "
            f"kalan %80: **%{_k['kalan']*100:.1f}** · fark "
            f"**%{_k['fark']*100:+.1f}**")
        st.caption(
            "Bulunan tek şey en ucuz uçtaki zayıf sinyal — o da t=1.85 ile "
            "anlamlılık eşiğinin altında. Desil 2'den 9'a kadar düzenli bir "
            "düşüş YOK; skor orta bantta ayrım yapamıyor. Ayrıca test "
            "HAYATTA KALMA YANLILIĞI taşıyor (iflas edenler listede yok) — "
            "yani gerçek sonuç bundan DAHA KÖTÜ olabilir.")
        st.caption(
            "Bu bir başarısızlık değil bir bulgudur: 'şu sayının üstü "
            "pahalı' demek yerine 'A, B'den daha pahalı' demeye geçtik. "
            "İkincisi doğru, birincisi uydurma olurdu.")
    ed = m.get("edgar") or {}
    if ed.get("durum") == "OK":
        st.caption(f"🏛️ **SEC EDGAR:** {ed['ozet']} · CIK {ed['cik']} — "
                   f"resmî 10-K çaprazı aktif.")
        fl = m.get("filings") or []
        if fl:
            st.caption("📄 Resmî dosyalamalar: " + " · ".join(
                f"[{f['form']} ({f['tarih']})]({f['url']})" for f in fl))
    elif ed.get("durum") not in ("KAPALI", None):
        st.caption("🏛️ **SEC EDGAR:** resmî veriye ulaşılamadı — analiz "
                   "yalnız Yahoo kaynağıyla; kritik karar öncesi 10-K'ya "
                   "kendin bak.")
    gec = m.get("gecmis") or []
    if gec:
        son = gec[-1]
        eski_f, simdi = son.get("fiyat"), m.get("price")
        if eski_f and simdi:
            d = simdi / eski_f - 1
            st.caption(f"📓 **Önceki analiz** ({son.get('tarih', '?')}): "
                       f"{son.get('hukum', '?')} @ {m['cur']}{eski_f:,.2f} "
                       f"→ bugün {m['cur']}{simdi:,.2f} ({d*100:+.1f}%) — "
                       f"Claude raporu bu kaydı kalibrasyon için kıyaslar.")
        else:
            st.caption(f"📓 Önceki analiz kaydı var "
                       f"({son.get('tarih', '?')}).")
    if m["ccy_mismatch"]:
        st.warning(f"Para birimi uyumsuzluğu: fiyat {m['ccy']}, tablolar "
                   f"{m['fin_ccy']}. Hisse başı DCF'i ihtiyatla oku.")


def render_valuation_tab(m, a, edu=False):
    # ── Denetim Bulgu 1b/5: üretilen ama HİÇ gösterilmeyen notlar ────
    # (iflas kesintisi sessizce değeri kırpıyordu; kapsama uyarısı ve
    # negatif-payda açıklaması çöpe gidiyordu). Artık ekranda.
    if (m.get("distres") or {}).get("oran"):
        st.warning(f"⚠️ **İFLAS KESİNTİSİ AKTİF** "
                   f"(−%{m['distres']['oran']*100:.1f}): "
                   f"{m['distres']['gerekce']} Aşağıdaki adil değer, bant, "
                   f"Monte Carlo, EPV ve duyarlılık matrisi bu kesintiyi "
                   f"İÇERİR.")
    if m.get("terminal_reinvest_notu"):
        st.caption("🧮 " + m["terminal_reinvest_notu"])
    if m.get("skor_kapsama_notu"):
        st.warning("📉 " + m["skor_kapsama_notu"])
    if m.get("negatif_payda_notu"):
        st.caption("ℹ️ " + m["negatif_payda_notu"])
    if m.get("pe_uyumsuzluk_notu"):
        st.caption("ℹ️ " + m["pe_uyumsuzluk_notu"])
    if edu:
        st.caption("📚 **Nasıl okunur:** *Adil değer*, varsayılan büyüme ve "
                   "iskontoyla gelecekteki nakitlerin bugünkü değeridir; ısı "
                   "haritası bu varsayımlar oynayınca değerin nasıl "
                   "değiştiğini gösterir. *Ters-DCF* soruyu tersine çevirir: "
                   "'bu fiyat kaç % büyüme satın alıyor?' — çıkan sayıyı "
                   "şirketin gerçek temposuyla kıyasla; makas büyükse fiyat "
                   "hikâye satıyordur.")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        fg = fig_gauge(m)
        if fg is not None:
            st.plotly_chart(fg, width="stretch")
            _ph = m.get("pahalilik_ham")
            _sk = m.get("score")
            if _sk is not None and _sk >= 90:
                _n = []
                _fr, _pr = m.get("fair"), m.get("price")
                _mk = (_pr / _fr) if (_fr and _pr and _fr > 0) else None
                if _mk and _mk > 1.5:
                    _n.append(f"fiyat, DCF adilinin ×{_mk:.1f} katı")
                _pe = m.get("pe_t")
                if _pe is None:
                    _n.append("F/K yok (kâr yok)")
                elif _pe > 35:
                    _n.append(f"F/K {_pe:.0f}")
                if m.get("negatif_payda_notu"):
                    _n.append("FCF− → EV/Satış cezası")
                _evs = m.get("ev_sales")
                if _evs and _evs > 8:
                    _n.append(f"EV/Satış {_evs:.0f}x")
                if m.get("flags_count", 0) >= 4 or len(m.get("flags") or []) >= 4:
                    _n.append("bayrak primi")
                _ek = (f" · HAM {_ph:.0f} → 97'ye kırpıldı"
                       if (_ph is not None and _ph > 97) else "")
                st.caption("🔎 Skoru uca sürükleyenler: "
                           + (" + ".join(_n) if _n else "bileşen dökümü yok")
                           + _ek + " — 97'ler arasında ayrım için bu satırı "
                           "ve HAM değeri kıyasla.")
        else:
            st.info("Skor üretilecek veri yok.")
    with c2:
        fs = fig_sensitivity(m)
        if fs:
            st.plotly_chart(fs, width="stretch")
            if (m.get("sens") or {}).get("distres_notu"):
                st.caption("⚠️ " + m["sens"]["distres_notu"])
        else:
            st.info("FCF bazı pozitif olmadığı için DCF/duyarlılık üretilmedi. "
                    "Bu başlı başına bir bulgudur: fiyat, henüz var olmayan "
                    "nakit akışına yaslanıyor.")

    if m.get("fair") is not None:
        scn = m.get("fair_scn") or {}
        p_, o_ = scn.get("kotumser", {}), scn.get("iyimser", {})
        s1, s2, s3 = st.columns(3)
        s1.metric("Kötümser", (f"{m['cur']}{m['fair_lo']:,.2f}"
                               if m.get("fair_lo") else "—"),
                  help=(f"Büyüme %{p_.get('g', 0)*100:.1f} · iskonto "
                        f"%{p_.get('d', 0)*100:.1f}"))
        s2.metric("Baz Senaryo", f"{m['cur']}{m['fair']:,.2f}",
                  help=(f"Büyüme %{a['growth']*100:.1f} · iskonto "
                        f"%{a['discount']*100:.1f} — kenar çubuğundaki "
                        f"varsayımlar"))
        s3.metric("İyimser", (f"{m['cur']}{m['fair_hi']:,.2f}"
                              if m.get("fair_hi") else "—"),
                  help=(f"Büyüme %{o_.get('g', 0)*100:.1f} · iskonto "
                        f"%{o_.get('d', 0)*100:.1f}"))
        st.caption("⚠️ Adil değer tek bir sayı DEĞİLDİR — varsayıma göre "
                   "değişen bir aralıktır. Karar verirken fiyatın bu "
                   "aralığın NERESİNDE durduğuna bak: aralığın üstündeyse "
                   "pahalı, altındaysa iskontolu bölgedesin.")
        # ── GİRDİ DUYARLILIĞI (rapor G6 — sahte kesinliğe karşı) ─────
        try:
            # Denetim Bulgu 3: motor artık roic_terminal kullanıyor —
            # eğitici duyarlılık da AYNI modelle hesaplanmalı, yoksa
            # gösterilen deltalar ekrandaki adil değerle kıyaslanamaz.
            # (render'a gelen 'a' compute-öncesi kopya olabilir; güncel
            # varsayımlar m["assumptions"]'ta.)
            _rt9 = (m.get("assumptions") or {}).get("roic_terminal")
            _dv1 = dcf_fair_value(m["fcf_base"], a["growth"],
                                  a["discount"] + 0.01, a["terminal"],
                                  a["years"], m["net_debt"], m["shares"],
                                  a["net_debt_adjust"],
                                  roic_terminal=_rt9)
            _dv2 = dcf_fair_value(m["fcf_base"], a["growth"],
                                  max(a["discount"] - 0.01,
                                      a["terminal"] + 0.005),
                                  a["terminal"], a["years"],
                                  m["net_debt"], m["shares"],
                                  a["net_debt_adjust"],
                                  roic_terminal=_rt9)
            _gv1 = dcf_fair_value(m["fcf_base"], a["growth"] + 0.01,
                                  a["discount"], a["terminal"],
                                  a["years"], m["net_debt"], m["shares"],
                                  a["net_debt_adjust"],
                                  roic_terminal=_rt9)
            _gv2 = dcf_fair_value(m["fcf_base"],
                                  max(a["growth"] - 0.01, -0.05),
                                  a["discount"], a["terminal"],
                                  a["years"], m["net_debt"], m["shares"],
                                  a["net_debt_adjust"],
                                  roic_terminal=_rt9)
            if None not in (_dv1, _dv2, _gv1, _gv2):
                st.caption(f"🎚️ Girdi duyarlılığı: iskonto ±1 puan → "
                           f"{m['cur']}{_dv1:,.0f}–{m['cur']}{_dv2:,.0f}"
                           f" · büyüme ±1 puan → "
                           f"{m['cur']}{_gv2:,.0f}–{m['cur']}{_gv1:,.0f}"
                           f" — girdi belirsizliği, çıktının gösterdiği "
                           f"hassasiyetten BÜYÜKTÜR (rapor: sahte "
                           f"kesinliğe karşı).")
        except Exception:
            pass
        # ── BEKLENTİ ÇIPASI (rapor G11): mutlak DCF'i ters-DCF ile ────
        # yan yana oku — hüküm değil, beklenti ölçüsü.
        if (m.get("implied_g") is not None
                and not m.get("implied_capped")):
            st.info(f"🧭 **Beklenti çıpası (rapor G11):** Mutlak DCF "
                    f"({m['cur']}{m['fair']:,.2f}) bir ucuz/pahalı HÜKMÜ "
                    f"değil, BEKLENTİ ölçüsüdür. Fiyat yıllık "
                    f"~{fmt_pct(m['implied_g'])} FCF büyümesi fiyatlıyor; "
                    f"motorun bu şirket için kullandığı tempo "
                    f"{fmt_pct(m.get('buyume_kullanilan'))}. Asıl soru "
                    f"'DCF ne diyor' değil, 'fiyata gömülü beklenti "
                    f"GERÇEKÇİ Mİ?' — mutlak DCF kaliteli şirketlere "
                    f"kronik 'pahalı' der (MRK/AAPL dersi); hükmü skor "
                    f"ve ters-DCF verir.")

    mc = m.get("mc")
    if mc:
        st.markdown("#### 🎲 Monte Carlo — Adil Değer Dağılımı")
        if edu:
            st.caption("📚 Varsayımlar tek sayı değil belirsizlik bandıdır; "
                       "10.000 senaryo çekildi. P5–P95 bandı DAR ise "
                       "değerleme sağlam, GENİŞ ise varsayıma aşırı bağımlı "
                       "demektir.")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("P5 (kötü)", f"{m['cur']}{mc['p5']:,.0f}")
        mc2.metric("Medyan (P50)", f"{m['cur']}{mc['p50']:,.0f}",
                   help=GLOSSARY["montecarlo"])
        mc3.metric("P95 (iyi)", f"{m['cur']}{mc['p95']:,.0f}")
        po = mc.get("yukari_potansiyel_olasiligi")
        mc4.metric("Adil > Fiyat Olasılığı",
                   f"%{po*100:.0f}" if po is not None else "—",
                   help="Senaryoların yüzde kaçında adil değer mevcut "
                        "fiyatın üzerinde çıktı.")
        dag = mc.get("dagilim") or []
        if dag:
            # B11-10: histogram rengi Wong gök, fiyat çizgisi Wong sarı
            # (11.22:1 kontrast) — koyu zeminde en okunur vurgu.
            fh = go.Figure(go.Histogram(x=dag, nbinsx=24,
                                        marker_color=WONG["gok"]))
            fh.add_vline(x=m["price"], line_dash="dash",
                         line_color=WONG["sari"], annotation_text="Fiyat")
            fh.update_layout(title="Adil değer dağılımı (10.000 senaryo)",
                             **_LAYOUT)
            fh.update_layout(height=260)
            st.plotly_chart(fh, width="stretch")

    epv = m.get("epv")
    if epv and epv.get("epv_hisse_basi"):
        st.markdown("#### 🏛️ EPV — Büyümesiz Taban Değer")
        if edu:
            st.caption("📚 EPV (Kazanç Gücü Değeri): Şirket hiç büyümese, "
                       "sadece bugünkü kazancını sonsuza dek sürdürse ne "
                       "eder? DCF 'iyimser', EPV 'taban'dır. Fiyat EPV'nin "
                       "çok üstündeyse, ödediğin her şey büyüme umuduna — "
                       "yani kırılgan.")
        e1, e2, e3 = st.columns(3)
        e1.metric("EPV / Hisse", f"{m['cur']}{epv['epv_hisse_basi']:,.2f}",
                  help=GLOSSARY.get("epv", ""))
        e2.metric("Normalize EBIT",
                  f"{m['cur']}{epv['normalize_ebit_musd']:,.0f}M")
        fg = epv.get("fiyata_gore")
        e3.metric("Fiyat / EPV Primi",
                  f"%{fg*100:+.0f}" if fg is not None else "—",
                  help="Fiyat EPV'nin ne kadar üstünde/altında. Yüksek "
                       "pozitif = değerin çoğu büyüme beklentisinde.")
        st.caption(epv.get("yorum", ""))
        if epv.get("bakim_capex_duzeltmesi"):
            st.caption("🏗️ EPV bakım-capex düzeltmesi: "
                       + epv["bakim_capex_duzeltmesi"]
                       + " — EPV amortismanı değil GERÇEK bakım harcamasını "
                         "gider sayar (Greenwald). Bu düzeltme yalnız EPV'de "
                         "yapılır; büyüme projeksiyonlu DCF'te yapılsaydı "
                         "çifte sayım olurdu.")
    elif epv and epv.get("uygulanamaz"):
        st.caption(f"🏛️ EPV: {epv['uygulanamaz']}")

    nz = m.get("norm")
    if nz and not nz.get("uygulanamaz"):
        st.markdown("#### 🧮 Normalize Kazanç (Orta-Döngü)")
        if edu:
            st.caption("📚 Tek yılın kârı yanıltır: zirvede 'ucuz', dipte "
                       "'pahalı' görünür. Medyan marjla normalize EPS, "
                       "döngünün ortasındaki gerçek kazanç gücüdür. SEC "
                       "serisi varsa 10 yıla kadar bakar.")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Medyan Marj", fmt_pct(nz["medyan_marj"]),
                  help=GLOSSARY.get("normkaz", ""))
        n2.metric("Güncel Marj", fmt_pct(nz["guncel_marj"]),
                  delta=f"{nz['sapma']*100:+.0f}% medyana göre")
        n3.metric("Normalize F/K",
                  fmt_x(nz.get("normalize_fk")),
                  help="Fiyat / normalize EPS — döngüden arındırılmış "
                       "gerçek çarpan.")
        n4.metric("Raporlanan F/K", fmt_x(nz.get("raporlanan_fk")))
        (st.warning if nz["durum"] == "ZİRVE ÜSTÜ"
         else st.info)(f"**{nz['durum']}** — {nz['yorum']}")
        if nz.get("normalize_taban_deger"):
            st.caption(f"Normalize taban değer: "
                       f"{m['cur']}{nz['normalize_taban_deger']:,.2f}/hisse "
                       f"(medyan marj × WACC, EPV'nin uzun-seri kardeşi) · "
                       f"kaynak: {nz['kaynak']}")
        else:
            st.caption(f"Kaynak: {nz['kaynak']}")
    elif nz and nz.get("uygulanamaz"):
        st.caption(f"🧮 Normalize kazanç: {nz['uygulanamaz']}")

    scn = m.get("scenario")
    if scn:
        st.markdown("#### 🎯 Senaryo-Olasılık Ağırlıklı Beklenen Değer")
        if edu:
            st.caption("📚 Monte Carlo'nun binlerce senaryosunu 3 net "
                       "senaryoya indirip olasılık veriyoruz. 'Beklenen "
                       "değer', bu üçünün olasılıkla ağırlıklı ortalaması — "
                       "fiyatla kıyaslanır.")
        sc = scn["senaryolar"]
        sd = pd.DataFrame([
            {"Senaryo": "🐻 Ayı", "Değer": f"{m['cur']}{sc['ayi']['deger']:,.0f}",
             "Olasılık": f"%{sc['ayi']['olasilik']*100:.0f}"},
            {"Senaryo": "⚖️ Baz", "Değer": f"{m['cur']}{sc['baz']['deger']:,.0f}",
             "Olasılık": f"%{sc['baz']['olasilik']*100:.0f}"},
            {"Senaryo": "🐂 Boğa", "Değer": f"{m['cur']}{sc['boga']['deger']:,.0f}",
             "Olasılık": f"%{sc['boga']['olasilik']*100:.0f}"}])
        st.dataframe(sd, hide_index=True, width="stretch")
        b1, b2, b3 = st.columns(3)
        b1.metric("Beklenen Değer", f"{m['cur']}{scn['beklenen_deger']:,.2f}")
        b2.metric("Beklenen Getiri", fmt_pct(scn["beklenen_getiri"]))
        asm = scn.get("yukari_asagi_asimetri")
        b3.metric("Yukarı/Aşağı Asimetri",
                  f"{asm:.2f}x" if asm else "—",
                  help="1'den büyükse yukarı potansiyel aşağı riskten fazla.")
        (st.success if scn["beklenen_getiri"] > 0.15
         else st.error if scn["beklenen_getiri"] < -0.15
         else st.info)(scn["hukum"])
        st.caption(scn.get("not", ""))

    ig = m.get("implied_g")
    if m.get("implied_capped"):
        if (ig or 0) > 0:
            st.error("**Ters-DCF sınırda:** Bu fiyatı haklı çıkarmak için "
                     "gereken FCF büyümesi, modelin üst sınırını (yılda %60) "
                     "bile aşıyor. Sayı vermek yanıltıcı olur — mesaj net: "
                     "**hiçbir gerçekçi büyüme senaryosu bu fiyatı "
                     "taşımıyor.** Fiyat, temelden kopmuş.")
        else:
            st.warning("**Ters-DCF sınırda:** Fiyat o kadar düşük ki model "
                       "alt sınıra dayandı — piyasa fiilen kalıcı küçülme "
                       "fiyatlıyor. Ya derin ucuzluk ya da piyasanın bildiği "
                       "bir şey var: falsifikasyon şart.")
    elif ig is not None:
        hist_g = m.get("fcf_cagr3")
        hist_txt = (f" Tarihsel 3Y FCF CAGR: **{fmt_pct(hist_g)}**."
                    if hist_g is not None else "")
        st.info(f"**Ters-DCF:** Mevcut fiyat, önümüzdeki {a['years']} yıl "
                f"boyunca yıllık **~{fmt_pct(ig)}** FCF büyümesini şart "
                f"koşuyor.{hist_txt}")
        if hist_g is not None and ig > max(hist_g * 1.5, hist_g + 0.05):
            st.warning("Piyasa, şirketin kendi geçmişinin belirgin üzerinde "
                       "bir hikâye fiyatlıyor — ispat yükü alıcıda.")

    fw = m.get("fwd")
    if fw:
        st.markdown("#### 🔭 İleriye Bakış & Tarihsel F/K Bandı")
        if edu:
            st.caption("📚 Solda analistlerin GELECEK kâr tahmini aralığı "
                       "(aralık genişse belirsizlik yüksek). Sağda bugünkü "
                       "F/K'nın kendi tarihindeki yeri — bandın üstü çoğu "
                       "zaman çarpan genişlemesidir (duygu), temel değil.")
        gy = fw.get("gelecek_yil") or fw.get("bu_yil") or {}
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Fwd EPS (ort.)",
                  f"{gy['ort']:.2f}" if gy.get("ort") else "—",
                  help=(f"{gy.get('analist', '?')} analist" if gy else None))
        f2.metric("Fwd EPS Aralığı",
                  (f"{gy['dusuk']:.2f}–{gy['yuksek']:.2f}"
                   if gy.get("dusuk") is not None
                   and gy.get("yuksek") is not None else "—"),
                  help="Analist düşük–yüksek tahmin bandı.")
        f3.metric("Forward F/K", fmt_x(fw.get("forward_fk")))
        f4.metric("Tahmin Belirsizliği", fw.get("belirsizlik", "—"),
                  help=GLOSSARY.get("fkband", ""))
        bnd = fw.get("fk_tarihsel_bant")
        if bnd:
            if bnd.get("hisse_kaynak_notu"):
                st.caption("⚠️ " + bnd["hisse_kaynak_notu"])   # O6 ekrana bağlandı
            kon = bnd.get("konum_0dip_1zirve")
            st.caption(f"Tarihsel F/K bandı ({len(bnd['seri'])} yıl): "
                       f"düşük {bnd['dusuk']}x · medyan {bnd['medyan']}x · "
                       f"yüksek {bnd['yuksek']}x"
                       + (f" — güncel konum: bandın "
                          f"%{kon*100:.0f}'inde"
                          + (" (bandın ÜSTÜNDE)" if kon > 1 else
                             " (dip yakını)" if kon < 0.2 else "")
                          if kon is not None else ""))

    br = m.get("baserate")
    if br:
        st.markdown("#### 🌍 Baz Oran — Dış Görüş")
        if edu:
            st.caption("📚 İçeriden bakış 'bu şirket özel' der; dış görüş "
                       "'benzerleri tarihsel olarak ne başardı?' diye sorar. "
                       "İkisi çatışıyorsa ispat yükü hissededir.")
        st.info(f"Hedef büyüme **%{br['hedef_buyume']*100:.0f}** → 10 yıl "
                f"sürdürme baz oranı: **{br['10y_surdurme_baz_orani']}**. "
                f"{br['mesaj']}")
        if br.get("gecmisle_kiyas"):
            st.caption(br["gecmisle_kiyas"])
        st.caption(br.get("not", ""))

    k1, k2, k3 = st.columns(3)
    k1.metric("52 Hafta Konumu", fmt_pct(m.get("pos_52w"), 0),
              help=GLOSSARY["pos52"])
    band = m.get("ps_band") or {}
    k2.metric("F/S Tarihsel Konum",
              fmt_pct(band.get("konum_0dip_1zirve"), 0),
              help=GLOSSARY["psband"])
    k3.metric("Fiyat/Gelir Makası (2Y)",
              f"{m['pv_gap']:.1f}x" if m.get("pv_gap") is not None else "—",
              help="2 yıllık fiyat değişiminin gelir değişimine oranı. "
                   "1x civarı sağlıklı; 3x+ ise fiyat temelden kopmuş, "
                   "getiriyi çarpan genişlemesi taşımış demektir.")

    st.markdown("#### Çarpanlar")
    ent = [
        ("F/K (trailing)", fmt_x(m["pe_t"]), "pe"),
        ("F/K (forward)", fmt_x(m["pe_f"]), "pe"),
        ("PEG", fmt_x(m["peg"])
         + (f"  ·  {m['peg_src']}" if m["peg_src"] else ""), "peg"),
        ("EV/EBITDA", fmt_x(m["ev_ebitda"]), "evebitda"),
        ("P/FCF (ham)", fmt_x(m["p_fcf"]), "pfcf"),
        ("P/FCF (SBC düzeltilmiş)", fmt_x(m["p_fcf_adj"]), "pfcf"),
        ("P/S", fmt_x(m["ps"]), "ps"),
        ("P/B", fmt_x(m["pb"]), "pb"),
    ]
    if edu:
        df = pd.DataFrame([(n, v, GLOSSARY[g]) for n, v, g in ent],
                          columns=["Çarpan", "Değer", "Bu şu demek"])
    else:
        df = pd.DataFrame([(n, v) for n, v, _ in ent],
                          columns=["Çarpan", "Değer"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"DCF bazı: {m['fcf_source']}"
               + (" · SBC düşüldü" if a["sbc_adjust"] else "")
               + (" · net borç düşüldü" if a["net_debt_adjust"] else ""))
    if m.get("dcf_akis_kimligi"):
        st.caption("ℹ️ " + m["dcf_akis_kimligi"])
    if m.get("ev_ebitda_note"):
        st.caption("ℹ️ EV/EBITDA: " + m["ev_ebitda_note"] + ".")
    if m.get("buyume_kaynak"):
        st.caption(f"ℹ️ DCF büyüme varsayımı: "
                   f"%{(m.get('buyume_kullanilan') or 0)*100:.1f} — "
                   f"{m['buyume_kaynak']}.")
    if m.get("ttm_kaynak"):
        st.caption("ℹ️ TTM kaynağı: " + m["ttm_kaynak"] + ".")
    if m.get("ev_kopru_notu"):
        st.caption("ℹ️ " + m["ev_kopru_notu"])
    if m.get("sbc_yontem"):
        st.caption("ℹ️ SBC: " + m["sbc_yontem"])
    if m.get("pe_t_kaynak"):
        st.caption("ℹ️ F/K (trailing): Yahoo alanı boştu — "
                   + m["pe_t_kaynak"] + " kullanıldı.")

    pr = m.get("peers")
    if pr:
        st.markdown("#### 👥 Rakip Kıyası")
        st.dataframe(pd.DataFrame(
            [{"Ticker": r["ticker"],
              "F/K (fwd)": fmt_x(r.get("pe_f")),
              "EV/EBITDA": fmt_x(r.get("ev_ebitda")),
              "F/S": fmt_x(r.get("ps")),
              "Brüt Marj": fmt_pct(r.get("brut_marj")),
              "Büyüme": fmt_pct(r.get("gelir_buyume")),
              "PD": fmt_money(r.get("mcap"), "$")}
             for r in pr["tablo"]]),
            hide_index=True, width="stretch")
        for g_ in pr.get("goreli") or []:
            st.caption("• " + g_)
        st.caption(pr.get("not", ""))

    kp = m.get("kademe")
    if kp:
        st.markdown("#### 🪜 Kademeli Giriş / Bekleme Planı")
        st.caption("⚠️ Tavsiye değil — DİSİPLİN çerçevesi. Nihai plan, "
                   "Claude raporunda katalizör ve falsifikasyonla rafine "
                   "edilir.")
        _sg = m.get("plan_saglama")
        if _sg:
            st.error("🚨 **SEVİYE SAĞLAMASI DÜŞTÜ — bu plana GÜVENME**\n\n"
                     + "\n".join(f"- {x}" for x in _sg)
                     + "\n\nSeviyeler tutarsız çıktı. Uydurma seviyeyle işlem "
                       "yapmaktansa hiç yapmamak doğrudur. Bu ekranı "
                       "kaydet ve bildir.")
        if kp["mod"] == "PLAN_YOK_IZLE":
            st.error(f"**{kp['mesaj']}**  İzleme eşiği: "
                     f"{m['cur']}{kp['izleme_esigi']:,.2f}")
            if kp.get("esik_kaynagi"):
                st.caption(f"Eşik nasıl bulundu: {kp['esik_kaynagi']}.")
            if kp.get("esik_taban_notu"):
                st.warning("🧱 " + kp["esik_taban_notu"])
            if kp.get("esik_hareketli_uyarisi"):
                st.warning("🎯 " + kp["esik_hareketli_uyarisi"])
            if kp.get("yapisal_baglam"):
                st.info("📐 " + kp["yapisal_baglam"])
            if kp.get("yontem_sinirlari"):
                with st.expander("⚖️ Bu hükmün kanıt sınırları"):
                    st.caption(kp["yontem_sinirlari"])
        else:
            st.info(kp["mesaj"])
            st.dataframe(pd.DataFrame(
                [{"Dilim": d["dilim"],
                  "Seviye": f"{m['cur']}{d['seviye']:,.2f}",
                  "Çapraz Teyit": d.get("teyit") or "—",
                  "Oran": d["oran"]} for d in kp["dilimler"]]),
                hide_index=True, width="stretch")
            if kp.get("cipa_notu"):
                st.caption(kp["cipa_notu"])
            if kp.get("yontem_durustlugu"):
                with st.expander("⚖️ Kademeli giriş neye mal olur?"):
                    st.caption(kp["yontem_durustlugu"])
        bag = kp.get("deger_baglami") or {}
        if any(bag.values()):
            parca = [f"DCF baz {m['cur']}{bag['dcf_baz']:,.2f}"
                     if bag.get("dcf_baz") else None,
                     f"MC P25 {m['cur']}{bag['mc_p25']:,.2f}"
                     if bag.get("mc_p25") else None,
                     f"MC P50 {m['cur']}{bag['mc_p50']:,.2f}"
                     if bag.get("mc_p50") else None]
            st.caption("🎯 Değer bağlamı (bağımsız çıpalar): "
                       + " · ".join(p for p in parca if p)
                       + " — seviyeleri bunlarla çapraz kontrol et.")
        for _t in kp.get("rejim_temposu") or []:
            st.warning("🌪️ " + _t)
        if kp.get("olay_notu"):
            st.warning(kp["olay_notu"])
        for k_ in kp.get("kurallar", []):
            st.caption("• " + k_)

    ck = m.get("cikis")
    if ck:
        st.markdown("#### 🚪 Kademeli Çıkış / Satış Planı")
        st.caption("⚠️ Tavsiye değil — girişin aynası olan DİSİPLİN "
                   "çerçevesi. Satış kararı maliyete değil DEĞERE ve TEZE "
                   "bağlanır.")
        if edu:
            st.caption("📚 Üç bacak: (1) değer hedefleri — fiyat oraya "
                       "gelince elindeki artık 'pahalı hisse'dir, kâr al; "
                       "(2) tez-bozan tetikler — fiyattan bağımsız çıkış; "
                       "(3) zaman stopu — ölü parayı taşıma.")
        st.info(f"**{ck['mod'].replace('_', ' ')}** — {ck['mesaj']}")
        poz = ck.get("pozisyon")
        rows = []
        for i, h in enumerate(ck["hedefler"]):
            r = {"Dilim": h["dilim"],
                 "Seviye": f"{m['cur']}{h['seviye']:,.2f}",
                 "Çıpa": h["kaynak"],
                 "Çapraz Teyit": h.get("teyit") or "—",
                 "Oran": h["oran"]}
            if poz:
                r["Maliyete Göre"] = fmt_pct(poz["dilim_kazanclari"][i])
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width="stretch")
        if ck.get("teyit_notu"):
            st.caption("🔎 " + ck["teyit_notu"])
        if any(h.get("not") for h in ck["hedefler"]):
            st.info("⬆️ " + next(h["not"] for h in ck["hedefler"] if h.get("not")))
        if ck.get("iki_sistem_notu"):
            st.caption("🎯 " + ck["iki_sistem_notu"])
        if poz:
            kz = poz["guncel_kz"]
            (st.success if kz >= 0 else st.error)(
                f"Pozisyon: maliyet {m['cur']}{poz['maliyet']:,.2f} → "
                f"güncel K/Z {fmt_pct(kz)}")
            st.warning(poz["uyari"])
        st.markdown("**Tez-bozan çıkış tetikleri (fiyattan bağımsız):**")
        for t_ in ck["tetikler"]:
            st.caption("• " + t_)
        st.caption("⏳ " + ck["zaman_stopu"])
        if ck.get("ufuk_notu"):
            st.warning("⏱️ " + ck["ufuk_notu"])
        for k_ in ck.get("kurallar", []):
            st.caption("• " + k_)

    uf = m.get("ufuk")
    if uf:
        st.markdown(f"#### ⏱️ Tutma Ufku Analizi — {uf['ufuk']}")
        u1, u2, u3 = st.columns(3)
        band = uf.get("tipik_bant_yuzde")
        _by = uf.get("bant_yukari_yuzde")
        _ba = uf.get("bant_asagi_yuzde")
        u1.metric("Tipik Dalgalanma (1σ)",
                  (f"+%{_by*100:.0f} / −%{_ba*100:.0f}"
                   if (_by is not None and _ba is not None)
                   else f"±%{band*100:.0f}" if band is not None else "—"),
                  help="Son 1 yılın gerçekleşen volatilitesi, ufuk "
                       "uzunluğuna ölçeklendi. Fiyatın bu bant içinde "
                       "gezinmesi NORMAL gürültüdür — tezle ilgisizdir.")
        md_ = uf.get("pencere_cekilme_medyan")
        kt_ = uf.get("pencere_cekilme_kotu")
        u2.metric("Tipik Pencere-İçi Çekilme",
                  fmt_pct(md_, 0) if md_ is not None else "—",
                  help=f"Son ~5 yılda bu uzunluktaki pencerelerin medyan en "
                       f"derin düşüşü ({uf.get('pencere_sayisi', 0)} "
                       f"pencere). 'Daha düşer mi?'nin dürüst ikamesi: bu "
                       f"hisse bu sürede TİPİK olarak bu kadar geri "
                       f"çekilir.")
        u3.metric("Kötü Pencere (p90)",
                  fmt_pct(kt_, 0) if kt_ is not None else "—",
                  help="Pencerelerin en kötü %10'unda görülen çekilme "
                       "derinliği — stres senaryosu referansı.")
        if uf.get("bant_alt") and uf.get("bant_ust"):
            st.caption(f"Bant: {m['cur']}{uf['bant_alt']:,.2f} – "
                       f"{m['cur']}{uf['bant_ust']:,.2f} · Zaman stopu: "
                       f"**{uf.get('zaman_stopu_tarihi', '—')}**")
        else:
            st.caption(f"Zaman stopu: **{uf.get('zaman_stopu_tarihi', '—')}"
                       "** · fiyat geçmişi yetersiz olduğu için bant "
                       "hesaplanamadı.")
        kz = uf.get("ufuk_katalizor") or []
        if kz:
            st.caption("📅 Ufka düşen olaylar: " + " · ".join(
                f"{e['olay']} ({e['tarih']}, {e['gun_kaldi']}g)"
                for e in kz)
                + (f" — pencerede **{uf.get('bilanco_sayisi', 0)} bilanço**"
                   f" var; kısa vadeli tezin asıl sınavı budur."
                   if uf.get("kisa_ufuk") else ""))
        elif uf.get("kisa_ufuk"):
            st.caption("📅 Ufukta takvimli katalizör görünmüyor — kısa "
                       "vadeli tez neyle gerçekleşecek? Bu soru cevapsızsa "
                       "pozisyonun gerekçesi zayıftır.")
        if uf.get("bant_notu"):
            st.caption("📐 " + uf["bant_notu"])
        if uf.get("dilim_tutarlilik"):
            st.caption("🪜 " + uf["dilim_tutarlilik"])
        st.info("🧭 " + uf.get("durustluk", ""))

    sr = m.get("sr")
    sp = m.get("stop")
    if sr or sp:
        st.markdown("#### 🛡️ Stop Planı")
        st.caption("Destek/direnç bölgelerinin TAM TABLOSU 🚩 Risk Radarı "
                   "sekmesine taşındı — burası yalnız stop ve pozisyon.")
    if sp and sp.get("durum") == "VERI_YOK":
        st.caption("🛡️ " + sp.get("not", ""))
    elif sp:
        s1, s2, s3 = st.columns(3)
        tr_ = sp.get("trailing") or {}
        tr_txt = (f"{m['cur']}{tr_['seviye']:,.2f}"
                  if tr_.get("seviye") is not None else "—")
        s1.metric("Chandelier Trailing (elde tutan)", tr_txt,
                  help=(tr_.get("formul", "") + " · "
                        + tr_.get("kural", "")))
        gs = sp.get("giris_stop") or {}
        gs_txt = (f"{m['cur']}{gs['seviye']:,.2f}"
                  if gs.get("seviye") is not None else "—")
        gs_delta = (f"-%{gs['mesafe_pct']*100:.0f} mesafe"
                    if gs.get("mesafe_pct") else None)
        s2.metric("Giriş Stopu (dilim-1 için)", gs_txt, gs_delta,
                  delta_color="off",
                  help=("Kaynak: " + gs.get("kaynak", "—") + ". "
                        + gs.get("not", "")))
        pz = sp.get("pozisyon") or {}
        hz = pz.get("hesap")
        # DENETİM BULGUSU N: hesap["risk_usd"] ve ["portfoy_pct"] üretiliyor
        # ama gösterilmiyordu — pozisyon boyutlandırmanın en önemli iki
        # sayısı (kaç dolar riske atıyorsun, portföyün yüzde kaçı) görünmez
        # kalıyordu. Eklendi.
        pz_txt = (f"{hz['hisse']} hisse (≈{m['cur']}{hz['pozisyon_usd']:,.0f})"
                  if hz else
                  (f"1.000$ risk ≈ {pz.get('ornek_1000usd_risk', '—')} "
                   f"hisse" if pz else "—"))
        s3.metric("Pozisyon (risk-bazlı)", pz_txt,
                  (f"risk {m['cur']}{hz['risk_usd']:,.0f} · portföyün "
                   f"%{hz['portfoy_pct']*100:.1f}'i"
                   if hz and hz.get("portfoy_pct") is not None else None),
                  delta_color="off",
                  help=(pz.get("kural", "") + " · " + pz.get("not", "")))
        if gs:
            yapi_txt = (f"{m['cur']}{gs['yapi_stop']:,.2f} (destek gücü "
                        f"{gs.get('destek_guc', '—')})"
                        if gs.get("yapi_stop") is not None else "—")
            if m.get("plan_saglama"):
                st.warning("🚨 Seviye sağlaması düştü — yukarıdaki uyarıyı "
                           "oku; bu stop/hedef sayılarına güvenme.")
            st.caption(f"Giriş stopu adayları → yapı: {yapi_txt} · "
                       f"{float((a or {}).get('giris_carpan') or 3.0):.1f}"
                       f"×ATR: {m['cur']}{gs['atr_stop']:,.2f} — GENİŞ "
                       f"olan seçildi (stop kümelenmesi + whipsaw "
                       f"savunması).")
        # DENETİM BULGUSU N: atr_pct, carpan ve destek_direnc'in "yontem"
        # alanı üretilip hiç gösterilmiyordu — hangi parametrelerle
        # hesaplandığı kullanıcıdan gizliydi (şeffaflık kaybı).
        _knt = [f"ATR22 {m['cur']}{sp['atr']:,.2f} "
                f"(fiyatın %{sp['atr_pct']*100:.1f}'i)",
                f"trailing çarpanı {sp['carpan']:.1f}×"]
        _yn = (m.get("sr") or {}).get("yontem")
        if _yn:
            _knt.append(_yn)
        st.caption("⚙️ Künye: " + " · ".join(_knt))
        if sp.get("atr_notu"):
            st.caption("ℹ️ " + sp["atr_notu"])
        st.warning("⚠️ " + sp.get("gap_notu", ""))
        st.caption("📉 " + sp.get("rejim_notu", ""))
        # DENETİM BULGUSU O: veri_tazeligi["fiyat_notu"] üretiliyor ama hiç
        # gösterilmiyordu. Kullanıcı stop tetiklerini bu fiyata göre kuruyor;
        # "~15 dk gecikmeli, gün içi tetikler gerçek zamanlı DEĞİL" uyarısının
        # görünmesi gereken yer tam burası.
        _fn = (m.get("tazelik") or {}).get("fiyat_notu")
        if _fn:
            st.caption("⏱️ " + _fn)

    # ═══ STOP TETİĞİ — canlı durum (kullanıcının istediği ayrı bölüm) ═══
    if sp and sp.get("durum") != "VERI_YOK":
        st.markdown("#### 🚨 Stop Tetiği — Şu Anki Durum")
        gs_ = (sp.get("giris_stop") or {})
        tr_ = (sp.get("trailing") or {})
        sv_giris, sv_trail = gs_.get("seviye"), tr_.get("seviye")
        fiyat = m.get("price")
        try:
            _h = m.get("_hist")
            son_kapanis = float(_h["Close"].dropna().iloc[-1])
            son_tarih = str(_h.index[-1].date())
        except Exception:
            son_kapanis, son_tarih = fiyat, "—"

        def _durum(sv, ad):
            if sv is None or son_kapanis is None:
                return None
            kirik = son_kapanis < sv
            mesafe = (son_kapanis / sv - 1) * 100
            return {"ad": ad, "seviye": sv, "kirik": kirik,
                    "mesafe": mesafe}

        satirlar = [x for x in (_durum(sv_giris, "Giriş stopu"),
                                _durum(sv_trail, "Trailing stop")) if x]
        if satirlar:
            cols = st.columns(len(satirlar))
            for i_, s_ in enumerate(satirlar):
                cols[i_].metric(
                    s_["ad"], f"{m['cur']}{s_['seviye']:,.2f}",
                    ("🔴 KIRILDI" if s_["kirik"]
                     else f"%{s_['mesafe']:.1f} yukarıda"),
                    delta_color="off")
        st.caption(f"Son günlük kapanış: **{m['cur']}{son_kapanis:,.2f}** "
                   f"({son_tarih}). **KURAL: gün içi fitil sayılmaz — "
                   f"tetik yalnız KAPANIŞ seviyenin altında olursa geçerlidir.**")
        for s_ in satirlar:
            if s_["kirik"]:
                st.error(f"🚨 {s_['ad']} KIRILDI — günlük kapanış "
                         f"{m['cur']}{son_kapanis:,.2f}, seviye "
                         f"{m['cur']}{s_['seviye']:,.2f}. Kural gereği "
                         f"çıkış değerlendirilmeli.")
        if not any(s_["kirik"] for s_ in satirlar):
            st.success("✅ Günlük kapanış tüm stop seviyelerinin üstünde — "
                       "tetik yok.")
        with st.expander("⏱️ 4 saatlik kapanışı da kontrol et"):
            st.caption("4 saatlik barlar Yahoo'nun saatlik verisinden "
                       "türetilir (~15 dk gecikmeli, ~2 yıl geriye). İşlem "
                       "saatleri dışında son bar eksik olabilir.")
            if st.button("4 saatlik kapanışı getir", key="s4h"):
                _h1 = fetch_saatlik(m.get("ticker") or "")
                _h4 = to_4h(_h1) if _h1 is not None else None
                if _h4 is None or _h4.empty:
                    st.info("4 saatlik veri alınamadı.")
                else:
                    k4 = float(_h4["Close"].iloc[-1])
                    t4 = str(_h4.index[-1])
                    st.metric("Son 4 saatlik kapanış",
                              f"{m['cur']}{k4:,.2f}", t4, delta_color="off")
                    for s_ in satirlar:
                        if k4 < s_["seviye"]:
                            st.error(f"🚨 4 saatlik kapanış {s_['ad']} "
                                     f"altında ({m['cur']}{s_['seviye']:,.2f})")
                        else:
                            st.success(f"✅ 4 saatlik kapanış {s_['ad']} "
                                       f"üstünde")

        rh = sp.get("r_hedef")
        if rh:
            st.markdown("**🎯 Kâr Alma Hedefi (doğrulanmış kural)**")
            h1, h2, h3 = st.columns(3)
            h1.metric("1R (risk)", f"{m['cur']}{rh['R']:,.2f}")
            h2.metric(f"Hedef ({rh['kat']:.0f}R)",
                      f"{m['cur']}{rh['seviye']:,.2f}",
                      (f"+%{rh['getiri_pct']*100:.0f}"
                       if rh.get("getiri_pct") else None), delta_color="off")
            tw = m.get("trend_ok")
            h3.metric("Trend (MA200)",
                      ("üstünde ✔" if tw else "ALTINDA ✖" if tw is False
                       else "—"),
                      (f"{m['cur']}{m['ma200']:,.2f}" if m.get("ma200")
                       else None), delta_color="off")
            st.caption("🎯 " + rh["not"])
            # DENETİM BULGUSU N: `yol_direnci` stop_plani'de ÜRETİLİYOR ama
            # arayüze HİÇ bağlanmamıştı — Claude'un kendi eklediği ölü değer,
            # eleştirdiği desenin aynısı. Bağlandı.
            _yd = rh.get("yol_direnci")
            if _yd:
                st.info(f"🧱 **Yol direnci:** {_yd['bant']} "
                        f"(güç {_yd['guc']:.0f}) — {_yd['not']}")
            if (a or {}).get("trend_filtre") and m.get("trend_ok") is False:
                st.warning("⚠️ Trend filtresi AÇIK ve fiyat MA200'ün "
                           "ALTINDA — bu kurulum filtreye takılıyor, "
                           "kural 'alma' diyor.")

    sen = m.get("seviye_senaryo")
    if sen:
        st.markdown("#### 🔀 Seviye Senaryoları — 'Kalırsa ne olur?'")
        st.caption("Soru: günlük kapanış şu seviyenin altında/üstünde "
                   "KALIRSA (2 ardışık kapanış) ne olur? Cevap TAHMİN "
                   "değil, bu hissenin KENDİ geçmişindeki benzer olayların "
                   "istatistiğidir. N küçükse sonuç çıkarma.")
        rows_s = []
        for r in sen:
            for yon_ad, blok in (("ALTINDA kalırsa",
                                  r.get("altina_kirilma")),
                                 ("ÜSTÜNDE kalırsa",
                                  r.get("ustune_gecis"))):
                uf10 = ((blok or {}).get("ufuklar") or {}).get("10g") or {}
                uf21 = ((blok or {}).get("ufuklar") or {}).get("21g") or {}
                rows_s.append({
                    "Seviye": f"{r.get('ad', '')} "
                              f"({m['cur']}{r['seviye']:,.2f})",
                    "Koşul": yon_ad,
                    "N": (blok or {}).get("n", 0),
                    "10g medyan": (f"%{uf10['medyan']*100:+.1f}"
                                   if uf10 else "—"),
                    "10g yön-devam": (f"%{uf10['yon_devam']*100:.0f}"
                                      if uf10 else "—"),
                    "21g medyan": (f"%{uf21['medyan']*100:+.1f}"
                                   if uf21 else "—"),
                    "21g geri-dönüş": (f"%{uf21['geri_donus']*100:.0f}"
                                       if uf21 else "—")})
        st.dataframe(pd.DataFrame(rows_s), hide_index=True,
                     width="stretch")
        st.caption("Okuma: *yön-devam* = ufuk sonunda kapanış olay "
                   "yönünde; *geri-dönüş* = pencere içinde seviyenin öbür "
                   "tarafına kapanış. Olasılık eğilimi — garanti değil.")
        with st.expander("🎛️ Kendi seviyeni test et (+ 4 saatlik)"):
            _vars = st.number_input(
                "Seviye", min_value=0.0,
                value=float(round(m.get("price") or 0.0, 2)),
                step=0.5, format="%.2f", key="sv_ozel")
            _tey = st.select_slider("Teyit (ardışık kapanış)",
                                    options=[1, 2, 3], value=2,
                                    key="sv_teyit")
            _k4 = st.toggle("4 saatlik de hesapla (≈2 yıl 1s verisinden; "
                            "~15 dk gecikme, örneklem SINIRLI)",
                            value=False, key="sv_4h")
            if st.button("Hesapla", key="sv_hesapla") and _vars > 0:
                def _sv_tab(res, etiket):
                    if not res:
                        st.info(f"{etiket}: yeterli veri yok.")
                        return
                    r2 = []
                    for yad, bl in (("ALTINDA kalırsa",
                                     res["altina_kirilma"]),
                                    ("ÜSTÜNDE kalırsa",
                                     res["ustune_gecis"])):
                        for hk, hv in (bl.get("ufuklar") or {}).items():
                            r2.append({"Koşul": yad, "Ufuk": hk,
                                       "N": hv["n"],
                                       "Medyan": f"%{hv['medyan']*100:+.1f}",
                                       "P25–P75":
                                       f"%{hv['p25']*100:+.1f} … "
                                       f"%{hv['p75']*100:+.1f}",
                                       "Yön-devam":
                                       f"%{hv['yon_devam']*100:.0f}",
                                       "Geri-dönüş":
                                       f"%{hv['geri_donus']*100:.0f}",
                                       "≥1×ATR derinleşme":
                                       f"%{hv['ek_derinlesme_1atr']*100:.0f}"
                                       })
                    st.markdown(f"**{etiket}** — seviye "
                                f"{m['cur']}{res['seviye']:,.2f}, teyit "
                                f"{res['teyit_kapanis']} kapanış")
                    if r2:
                        st.dataframe(pd.DataFrame(r2), hide_index=True,
                                     width="stretch")
                    else:
                        st.info("Bu seviyede tarihsel olay oluşmamış "
                                "(fiyat hiç bu tarafa geçmemiş olabilir).")
                _sv_tab(seviye_kosullu_istatistik(
                    m.get("_hist"), _vars, m.get("atr22"), _tey),
                    "GÜNLÜK (5 yıl)")
                if _k4:
                    _h1 = fetch_saatlik(m.get("ticker") or "")
                    _h4 = to_4h(_h1) if _h1 is not None else None
                    if _h4 is None:
                        st.info("4 saatlik veri alınamadı (Yahoo saatlik "
                                "sınırı/blok).")
                    else:
                        _a4s = atr_serisi(_h4, 22)
                        _a4 = (float(_a4s.iloc[-1])
                               if _a4s is not None and len(_a4s) else None)
                        _sv_tab(seviye_kosullu_istatistik(
                            _h4, _vars, _a4, _tey),
                            "4 SAATLİK (~2 yıl — örneklem sınırlı, "
                            "temkinli oku)")
                st.caption("🔀 Tarihsel baz oran — geleceği göstermez; "
                           "ÖNEMLİ YERLERİ ve o yerlerde geçmişte ne "
                           "olduğunu gösterir. Karar ve risk sende.")


def render_financials_tab(m, b, edu=False):
    if edu:
        st.caption("📚 **Nasıl okunur:** Gelir işin motoru, net kâr muhasebe "
                   "sonucu, FCF ise kasaya gerçekten giren nakittir — üçü "
                   "birlikte büyümeli. Marjlarda yön trendi seviyeden "
                   "önemlidir: genişleyen marj fiyatlama gücü, eriyen marj "
                   "rekabet baskısı demektir.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gelir CAGR (3Y)", fmt_pct(m["rev_cagr3"]),
              help=GLOSSARY["rvcagr"])
    k2.metric("FCF CAGR (3Y)", fmt_pct(m["fcf_cagr3"]),
              help=GLOSSARY["fcfcagr"])
    k3.metric("ROE", fmt_pct(m["roe"]), help=GLOSSARY["roe"])
    k4.metric("ROIC (~)", fmt_pct(m["roic"]),
              ("Ar-Ge düzeltmeli" if m.get("roic_arge") else None),
              delta_color="off", help=GLOSSARY["roic"])
    _ra = m.get("roic_arge")
    if _ra:
        st.warning(
            f"🔬 **Ar-Ge kapitalizasyonu uygulandı** (vergi etkisi Damodaran "
            f"Böl.9'a göre: düzeltme NOPAT'a ÖN-VERGİ eklenir) — ham ROIC "
            f"%{_ra['roic_ham']*100:.1f} → düzeltilmiş "
            f"%{_ra['roic_duzeltmeli']*100:.1f} "
            f"(**{_ra['fark_puan']:+.1f} puan**). Ar-Ge varlığı "
            f"${_ra['arge_varligi_musd']:,.0f}M, yıllık amortisman "
            f"${_ra['yillik_amortisman_musd']:,.0f}M, ömür "
            f"{_ra['etkin_omur_yil']} yıl. Muhasebe Ar-Ge'yi gider yazar "
            f"ama ekonomik olarak yatırımdır; düzeltilmemiş ROIC "
            f"Ar-Ge-yoğun şirketlerde ŞİŞER. Karar skorunda ve DCF büyüme "
            f"tavanında **düzeltilmiş** değer kullanıldı.")
        if _ra.get("omur_notu"):
            st.caption("⏳ " + _ra["omur_notu"])
    elif m.get("roic_kaynak"):
        st.caption("🔬 " + m["roic_kaynak"])
    if m.get("vergi_kaynak") and m.get("eff_tax_defter") is not None:
        _fark = (m.get("eff_tax", 0) - m["eff_tax_defter"]) * 100
        st.caption(f"🧾 **Vergi:** NOPAT/ROIC'te %{m.get('eff_tax',0)*100:.1f} "
                   f"kullanıldı — kaynak: {m['vergi_kaynak']} "
                   f"(defter oranı %{m['eff_tax_defter']*100:.1f}, "
                   f"fark {_fark:+.1f} puan). Ertelenmiş vergi bugün nakit "
                   f"çıkışı yaratmaz; Damodaran ödenen vergiyi önerir.")
    _rk = m.get("roic_kira_detay")
    if _rk:
        st.caption(f"🏢 ROIC kira kapitalizasyonu (Damodaran, ASC 842): "
                   f"kira yükümlülüğü "
                   f"${_rk['kira_yukumlulugu_musd']:,.0f}M yatırılan "
                   f"sermayeye, gömülü faiz "
                   f"${_rk['gomulu_faiz_musd']:,.0f}M (borç maliyeti "
                   f"%{_rk['borc_maliyeti']*100:.1f}) EBIT'e eklendi "
                   f"[TÜRETİLMİŞ ~].")
    _ga = m.get("geri_alim")
    if _ga:
        _tp = _ga.get("telafi_payi")
        (st.warning if (_tp or 0) >= 0.6 or _tp is None else st.caption)(
            f"🔁 **Geri alım kalitesi:** SBC ${_ga['sbc_musd']:,.0f}M · "
            f"geri alım ${_ga['geri_alim_musd']:,.0f}M → SBC telafisi "
            f"${_ga['sbc_telafisi_musd']:,.0f}M, **gerçek hissedar iadesi "
            f"${_ga['gercek_iade_musd']:,.0f}M** "
            + (f"(geri alımın %{_tp*100:.0f}'i telafi). " if _tp is not None
               else "(geri alım yok). ")
            + _ga["hukum"])
    _bc = m.get("capex_ayrim")
    if _bc:
        st.caption(
            f"🏗️ **Capex ayrımı ({_bc['donem']}):** toplam "
            f"${_bc['toplam_capex_musd']:,.0f}M = bakım "
            f"${_bc['bakim_capex_musd']:,.0f}M + büyüme "
            f"${_bc['buyume_capex_musd']:,.0f}M "
            f"(büyüme payı %{_bc['buyume_payi']*100:.0f}). "
            f"{_bc['gerekce']}. " + _bc["not"])
    if m["years"]:
        st.plotly_chart(fig_fundamentals(m), width="stretch")
        st.plotly_chart(fig_margins(m), width="stretch")
    else:
        st.info("Yıllık tablo verisi gelmedi (yfinance bu ticker için boş "
                "döndü olabilir).")
    if b["hist"] is not None:
        st.plotly_chart(fig_price(b["hist"], m["ticker"]),
                        width="stretch")

    dp = m.get("dupont")
    if dp:
        st.markdown("#### 🔬 DuPont — ROE'nin Sürücüleri")
        if edu:
            st.caption("📚 Aynı ROE üç farklı yoldan gelebilir: yüksek marj, "
                       "hızlı varlık devri, ya da bol kaldıraç (borç). "
                       "DuPont hangisinin sürüklediğini gösterir — "
                       "kaldıraçla şişmiş ROE kırılgandır.")
        d1, d2, d3 = st.columns(3)
        d1.metric("Net Marj", fmt_pct(dp["net_marj"]))
        d2.metric("Varlık Devri", f"{dp['varlik_devri']:.2f}x")
        d3.metric("Finansal Kaldıraç", f"{dp['kaldirac']:.2f}x")
        # DENETİM BULGUSU O: dupont 5-FAKTÖR ayrıştırmasını hesaplıyor
        # (vergi yükü × faiz yükü × EBIT marjı × devir × kaldıraç) ama
        # arayüz yalnız 3 faktörü gösteriyordu. 5 faktör, ROE'yi borç ve
        # vergi etkisinden arındırıp işletme performansını ayırır.
        _bf = dp.get("bes_faktor")
        if _bf:
            with st.expander("🔬 DuPont 5 faktör — ROE'nin tam ayrıştırması"):
                st.dataframe(pd.DataFrame([
                    {"Faktör": "Vergi yükü (NK/VÖK)",
                     "Değer": _bf.get("vergi_yuku"),
                     "Okuma": "1'e yakın = düşük efektif vergi"},
                    {"Faktör": "Faiz yükü (VÖK/EBIT)",
                     "Değer": _bf.get("faiz_yuku"),
                     "Okuma": "1'den uzak = faiz kârı yiyor"},
                    {"Faktör": "EBIT marjı", "Değer": _bf.get("ebit_marj"),
                     "Okuma": "işletme kârlılığı — asıl performans"},
                    {"Faktör": "Varlık devri",
                     "Değer": _bf.get("varlik_devri"),
                     "Okuma": "sermaye verimliliği"},
                    {"Faktör": "Kaldıraç", "Değer": _bf.get("kaldirac"),
                     "Okuma": "3x üstü ROE'yi borç şişiriyor"}]),
                    width="stretch", hide_index=True)
                st.caption("Beşinin ÇARPIMI ROE'yi verir. Yüksek ROE'nin "
                           "kaynağı EBIT marjı ve devirse sağlam; faiz/vergi "
                           "yükü ve kaldıraçsa kırılgandır.")
        for s_ in dp.get("surucu_notu", []):
            st.caption("• " + s_)

    ccc = m.get("ccc")
    if ccc and ccc.get("seri"):
        st.markdown("#### 💵 Nakit Dönüşüm Döngüsü (CCC)")
        if edu:
            st.caption("📚 Ürünü nakde çevirmek kaç gün sürüyor? Düşük/negatif "
                       "iyidir (tedarikçi parasıyla dönüyorsun). Artıyorsa "
                       "büyüme nakit yakıyor demektir.")
        yrs = sorted(ccc["seri"])
        cd = pd.DataFrame([
            {"Yıl": y, "DSO (tahsilat)": ccc["seri"][y].get("dso"),
             "DIO (stok)": ccc["seri"][y].get("dio"),
             "DPO (ödeme)": ccc["seri"][y].get("dpo"),
             "CCC (gün)": ccc["seri"][y].get("ccc")} for y in yrs])
        st.dataframe(cd, hide_index=True, width="stretch")
        if ccc.get("trend"):
            st.caption(f"Trend: {ccc['trend']}")

    tpl = m.get("sector_tpl")
    if tpl:
        st.markdown(f"#### 🏭 Sektör Şablonu — {tpl['tur']}")
        if edu:
            st.caption("📚 Her sektör kendi merceğiyle okunur: SaaS'ta Rule "
                       "of 40, bankada ROTCE/P-TBV, çevrimselde normalize "
                       "F/K. Genel metrikler burada yanıltabilir.")
        met = {k: v for k, v in (tpl.get("metrikler") or {}).items()
               if v is not None}
        if met:
            cols_ = st.columns(min(len(met), 4))
            for i, (k, v) in enumerate(list(met.items())[:4]):
                ad = k.replace("_", " ").title()
                if isinstance(v, float) and abs(v) < 5:
                    disp = (f"%{v*100:.0f}" if abs(v) <= 1.5
                            else f"{v:.2f}")
                else:
                    disp = f"{v}"
                cols_[i % len(cols_)].metric(ad, disp,
                    help=GLOSSARY.get("rule40") if "rule" in k
                    else GLOSSARY.get("rotce") if "rotce" in k else None)
        if tpl.get("yorum"):
            st.info(tpl["yorum"])
        kim = tpl.get("kimlik_kontrol")
        if kim:
            st.caption(f"Kimlik kontrol: P/TBV {kim['p_tbv']} vs F/K×ROTCE "
                       f"{kim['fk_x_rotce']} — {kim['not']}")
        for g_ in tpl.get("claude_gorevleri") or []:
            st.caption("🌐 Claude görevi: " + g_)
        if tpl.get("not"):
            st.caption(tpl["not"])

    ca = m.get("capalloc")
    if ca:
        st.markdown("#### 🏦 Sermaye Tahsisi")
        if edu:
            st.caption("📚 CEO'nun asıl işi: kazanılan parayı nereye "
                       "koyduğu. Tutulan her 1$ kâr ≥ 1$ piyasa değeri "
                       "yaratmalı (Buffett testi).")
        c1_, c2_ = st.columns([1, 2])
        c1_.metric("Tahsis Skoru", f"{ca['skor']:.0f} / 100",
                   help=GLOSSARY["sermaye"])
        c2_.markdown(f"**{ca['profil']}**")
        det = ca.get("detay") or {}
        bt = det.get("buffett_testi")
        if bt:
            st.caption(f"Buffett testi ({bt['donem']}): tutulan 1$ kâr → "
                       f"**{bt['deger_yaratma_orani']}$** piyasa değeri. "
                       f"{bt['not']}")
        dag = det.get("dagitim")
        if dag:
            st.caption(f"Dağıtım (3Y): temettü+geri alım "
                       f"${dag['temettu_geri_alim_3y_musd']:,.0f}M / FCF "
                       f"${dag['fcf_3y_musd']:,.0f}M → karşılama "
                       f"{dag['fcf_karsilama']}x"
                       + (f" ⚠️ {dag['uyari']}" if dag.get("uyari") else ""))
        rs = det.get("roic_seri")
        if rs:
            st.caption("ROIC serisi: " + " · ".join(
                f"{y}: %{v*100:.0f}" for y, v in sorted(rs.items()))
                + (f" ({det.get('roic_trend', '')})"
                   if det.get("roic_trend") else ""))
        if det.get("satin_alma"):
            st.warning(det["satin_alma"])
        if det.get("hisse_yonu"):
            st.caption(f"Hisse yönü: {det['hisse_yonu']}")
        st.caption(ca.get("not", ""))


def render_sr_tablosu(m):
    """DESTEK / DİRENÇ BÖLGELERİ — az ama güçlü. 1 günlük / 4 saatlik."""
    st.markdown("#### 📐 Destek / Direnç Bölgeleri")
    c1, c2, c3 = st.columns([2, 2, 3])
    periyot = c1.radio("Zaman dilimi", ["1 günlük", "4 saatlik"],
                       horizontal=True, key="sr_periyot")
    kalite = c2.select_slider(
        "Bölge kalitesi", options=["Gevşek", "Normal", "Sıkı"],
        value="Normal", key="sr_kalite",
        help="Sıkı = yalnız çok test edilmiş, hacimli, taze bölgeler. "
             "Gevşek = daha çok bölge ama zayıfları da girer.")
    esik = {"Gevşek": (40, 2, 5), "Normal": (55, 2, 3),
            "Sıkı": (70, 3, 2)}[kalite]
    c3.caption(f"Filtre: güç ≥ {esik[0]} · dokunma ≥ {esik[1]} · "
               f"en fazla {esik[2]} bölge/yön")
    with st.expander("⚖️ Bu seviyelerin kanıt durumu (oku)"):
        st.caption(
            "En güçlü akademik kanıt DÖVİZ ve GÜN-İÇİ veridedir (Osler "
            "2000/2003; mekanizma emir kümelenmesidir). HİSSE + 3-4 aylık "
            "ufuk için doğrudan kanıt YOKTUR. 'Dokunma sayısı arttıkça "
            "seviye güçlenir' iddiasının hakemli dayanağı bulunamadı — "
            "pratisyen geleneğidir ve buradaki güç skorunun bir bileşenidir. "
            "Sullivan-Timmermann-White (1999) veri-tarama düzeltmesi sonrası "
            "teknik kural üstünlüğünün örnek-dışı dönemde kaybolduğunu "
            "gösterdi. Bu bölgeleri KEHANET değil, kararı önceden yazmaya "
            "yarayan ÇIPA olarak kullan; asıl faydası fiyat oraya geldiğinde "
            "doğaçlama yapmanı engellemesidir.")

    if periyot == "4 saatlik":
        _h1 = fetch_saatlik(m.get("ticker") or "")
        _h4 = to_4h(_h1) if _h1 is not None else None
        if _h4 is None or len(_h4) < 60:
            st.warning("4 saatlik veri alınamadı (Yahoo saatlik sınırı ya "
                       "da işlem saati dışı). 1 günlüğe dön.")
            return
        _a4s = atr_serisi(_h4, 22)
        _a4 = (float(_a4s.iloc[-1]) if _a4s is not None and len(_a4s)
               else None)
        sr = destek_direnc(_h4, m.get("price"), _a4, k=5,
                           min_guc=esik[0], min_dokunma=esik[1],
                           maks_bolge=esik[2])
        st.caption("⏱️ 4 saatlik barlar Yahoo'nun saatlik verisinden "
                   "türetilir: ~2 yıl geriye gider, ~15 dk gecikmelidir. "
                   "Örneklem günlüğe göre SINIRLIDIR.")
    else:
        sr = destek_direnc(m.get("_hist"), m.get("price"), m.get("atr22"),
                           k=5, min_guc=esik[0], min_dokunma=esik[1],
                           maks_bolge=esik[2])
    if not sr:
        st.caption("Bu filtrede bölge kalmadı — kaliteyi 'Gevşek' yap ya "
                   "da fiyat geçmişi yetersiz.")
        return
    st.caption("Kanıta dayalı yöntemler: swing bölgeleri "
               "[Brock-Lakonishok-LeBaron 1992], 52h dip/zirve "
               "[George-Hwang 2004], yuvarlak sayı kümelenmesi [Osler]. "
               "Fibonacci/pivot bilinçli olarak YOK (kanıt: folklor). "
               "Seviyeler ÇİZGİ değil BÖLGEdir; yön garantisi vermez.")
    fiyat = m.get("price")
    rows_sr = []
    for tur, liste in (("Direnç", (sr.get("direncler") or [])[::-1]),
                       ("Destek", sr.get("destekler") or [])):
        for z in liste:
            orta = (z["bant_alt"] + z["bant_ust"]) / 2
            uz = ((orta / fiyat - 1) * 100) if fiyat else None
            rows_sr.append({
                "Tür": tur,
                "Bölge": (f"{m['cur']}{z['bant_alt']:,.2f}–"
                          f"{z['bant_ust']:,.2f}"),
                "Fiyata uzaklık": (f"%{uz:+.1f}" if uz is not None else "—"),
                "Güç": f"{z['guc']:.0f}/100",
                "Dokunma": z["dokunma"],
                "Son Test": f"{z['son_test_gun']}g önce",
                # DENETİM BULGUSU N: hacim_orani destek_direnc'te hesaplanıp
                # güç skoruna giriyor ama tabloda GÖSTERİLMİYORDU — kullanıcı
                # bölgenin neden güçlü olduğunu göremiyordu. Eklendi.
                "Hacim": (f"{z['hacim_orani']:.2f}x"
                          if z.get("hacim_orani") else "—"),
                "Etiketler": ", ".join(z.get("etiketler") or []) or "—"})
    if not rows_sr:
        st.caption("Bu hissede tanımlı bölge oluşmadı.")
        return
    st.dataframe(pd.DataFrame(rows_sr), hide_index=True,
                 width="stretch")
    if fiyat:
        st.caption(f"Güncel fiyat: **{m['cur']}{fiyat:,.2f}** — tablo "
                   f"yukarıdan aşağıya dirençler → destekler sırasında.")
    st.caption("🎯 " + (sr.get("not") or ""))
    st.caption("Güç skoru [TÜRETİLMİŞ]: dokunma sayısı + dokunma hacmi + "
               "tazelik + polarity. Yüksek güç 'tutar' demek değildir; "
               "geçmişte daha çok tepki görmüş demektir.")


def render_risk_tab(m, edu=False):
    render_sr_tablosu(m)
    st.divider()
    if edu:
        st.caption("📚 **Nasıl okunur:** Buradaki testler şirketin "
                   "SAĞLIĞINI ölçer, fiyatını değil. Piotroski genel "
                   "kondisyon, Altman iflas mesafesi, tahakkuk oranı kârın "
                   "nakit karşılığıdır. Bayraklar kural motorunun yakaladığı "
                   "somut anomalilerdir — her bayrak bir soru işaretidir, "
                   "idam fermanı değil; ama cevapsız bayrak biriktirme.")
    p = m["piotroski"]
    k1, k2, k3 = st.columns(3)
    # DENETİM BULGUSU O: piotroski_f geçen kriterlerin LİSTESİNİ üretiyor
    # ama arayüz yalnız "5/9" sayısını gösteriyordu. Asıl bilgi HANGİ 4
    # kriterin kaldığıdır — 9 kriter eşit ağırlıklı değildir okuyucu için.
    k1.metric("Piotroski F", f"{p['score']} / {p['max'] or '—'}",
              help=GLOSSARY["piotroski"])
    k2.metric("Altman Z",
              fmt_f(m["altman"]) if not m["is_financial"] else "n/a (finansal)",
              help=GLOSSARY["altman"])
    k3.metric("Tahakkuk Oranı", fmt_pct(m["accrual"]),
              help=GLOSSARY["accrual"])
    # DENETİM BULGUSU O: (1) piotroski_f geçen kriterlerin LİSTESİNİ
    # üretiyor ama arayüz yalnız "5/9" sayısını gösteriyordu — asıl bilgi
    # HANGİ kriterlerin kaldığıdır. (2) m["piotroski_notu"] (büyük-cap'te
    # F-Score'un seçim kriteri değil yalnız kırmızı bayrak olduğu uyarısı)
    # HİÇ gösterilmiyordu — yorumun tamamı buna bağlı.
    if p.get("passed") is not None and p.get("max"):
        with st.expander(f"🔬 Piotroski dökümü — {p['score']}/{p['max']} kriter"):
            _tum = ["ROA > 0", "CFO > 0", "ΔROA > 0",
                    "CFO > Net Kâr (tahakkuk)", "Kaldıraç ↓", "Cari oran ↑",
                    "Hisse sayısı artmadı", "Brüt marj ↑", "Varlık devri ↑"]
            _g = set(p["passed"])
            st.dataframe(pd.DataFrame(
                [{"Kriter": t, "Sonuç": "✔ geçti" if t in _g else "✖ kaldı"}
                 for t in _tum]), width="stretch", hide_index=True)
            st.caption("Veri eksik kriterler hiç sorulmaz → payda 9'dan "
                       "küçük olabilir. KALAN kriterler şirketin hangi "
                       "eksende zayıfladığını söyler.")
    if m.get("piotroski_notu"):
        st.caption("ℹ️ " + m["piotroski_notu"])
    k4, k5, k6 = st.columns(3)
    k4.metric("Net Borç / EBITDA", fmt_x(m["nd_ebitda"]),
              help=GLOSSARY["ndebitda"])
    k5.metric("Faiz Karşılama", fmt_x(m["int_cov"]),
              help=GLOSSARY["intcov"])
    k6.metric("Cari Oran", fmt_f(m["current_ratio"]),
              help=GLOSSARY["cari"])

    bm = m.get("beneish") or {}
    b1, b2c, b3c = st.columns(3)
    b1.metric("Beneish M-Skoru",
              (f"{bm['skor']:.2f}" if bm.get("skor") is not None
               else "n/a (finansal)" if m["is_financial"] else "—"),
              help=GLOSSARY["beneish"])
    b2c.metric("Manipülasyon Bölgesi", bm.get("bolge", "—"),
               help="TEMİZ < −2.22 < GRİ < −1.78 < RİSKLİ")
    b3c.metric("Dönem", bm.get("donem", "—"),
               help="Skorun kıyasladığı iki mali yıl.")
    # DENETİM BULGUSU O: beneish_m 8 bileşeni hesaplıyor ve Claude paketine
    # gönderiyor ama ARAYÜZDE hiç göstermiyordu. "RİSKLİ" bayrağını gören
    # kullanıcı HANGİ bileşenin skoru sürüklediğini göremiyordu — oysa
    # yorumun tamamı buna bağlı (DSRI alacak şişmesi, TATA tahakkuk, SGI
    # agresif büyüme...). Eklendi.
    if bm.get("bilesenler"):
        with st.expander("🔬 Beneish bileşenleri — skoru ne sürüklüyor?"):
            _ac = {"DSRI": "Alacak/satış oranı (gelir tanıma şişmesi)",
                   "GMI": "Brüt marj bozulması", "AQI": "Varlık kalitesi",
                   "SGI": "Satış büyüme hızı (agresif büyüme baskısı)",
                   "DEPI": "Amortisman yavaşlatma", "SGAI": "SG&A verimsizliği",
                   "TATA": "Toplam tahakkuk (kâr nakde dönmüyor)",
                   "LVGI": "Kaldıraç artışı"}
            _b = bm["bilesenler"]
            # nötr 1.0'dan sapma büyüdükçe katkı büyür (TATA katsayısı ayrı)
            _sap = sorted(((k, v, abs(v - (0 if k == "TATA" else 1.0)))
                           for k, v in _b.items()), key=lambda x: -x[2])
            st.dataframe(pd.DataFrame(
                [{"Bileşen": k, "Değer": v,
                  "Nötrden sapma": round(d, 3),
                  "Ne ölçüyor": _ac.get(k, "")} for k, v, d in _sap]),
                width="stretch", hide_index=True)
            st.caption(f"{bm.get('esikler','')} · dönem {bm.get('donem','')}. "
                       f"Sapması en büyük bileşen skoru en çok sürükleyendir; "
                       f"adli incelemeye ORADAN başla.")
    if bm.get("eksik_notr_alinan"):
        st.caption(f"Beneish: eksik bileşen(ler) nötr (1.0) alındı → "
                   f"{', '.join(bm['eksik_notr_alinan'])}")

    trap = m.get("trap") or {}
    if trap.get("uyarilar"):
        st.markdown("#### ⚠️ Tuzak Uyarıları")
        for u in trap["uyarilar"]:
            st.warning(u)
    elif trap.get("cevrimsel_mi"):
        st.caption("ℹ️ Çevrimsel sektör tespit edildi ama şu an aktif "
                   "çevrimsel/değer tuzağı sinyali yok.")

    # ── DURUM ŞERİDİ: renk + İKON + metin (Wong paleti, renk körü-güvenli)
    # rozet() bu güne kadar tanımlıydı ama HİÇ ÇAĞRILMIYORDU; durum
    # göstergeleri yalnız renkle kodlanıyordu. Erkeklerin ~%8'i kırmızı-
    # yeşil ayırt edemez — ikon olmadan bu bilgi onlar için kayıptır.
    _rz = []
    _pf = m.get("piotroski") or {}
    if _pf.get("max"):
        _o = _pf["score"] / _pf["max"]
        _rz.append(rozet("iyi" if _o >= 0.7 else "orta" if _o >= 0.45
                         else "kotu",
                         f"Piotroski {_pf['score']}/{_pf['max']}"))
    _al = _num(m.get("altman"))
    if _al is not None:
        _e = ALTMAN_BOLGE.get(m.get("altman_model") or "uretim",
                              ALTMAN_BOLGE["uretim"])
        _rz.append(rozet("kotu" if _al < _e["tehlike"] else
                         "orta" if _al < _e["guvenli"] else "iyi",
                         f"Altman {_al:.2f}"))
    elif m.get("is_financial"):
        _rz.append(rozet("bilinmiyor", "Altman n/a (finansal)"))
    _bn = m.get("beneish") or {}
    if _bn.get("bolge"):
        _rz.append(rozet({"TEMİZ": "iyi", "GRİ": "orta",
                          "RİSKLİ": "kotu"}.get(_bn["bolge"], "notr"),
                         f"Beneish {_bn['bolge']}"))
    _ac = _num(m.get("accrual"))
    if _ac is not None:
        _rz.append(rozet("iyi" if _ac < 0 else "orta" if _ac <= 0.05
                         else "kotu", f"Tahakkuk {fmt_pct(_ac)}"))
    if _rz:
        st.markdown(" &nbsp;·&nbsp; ".join(_rz), unsafe_allow_html=True)
        st.caption("Durum şeridi ikon + renk birlikte kodlanmıştır — "
                   "siyah-beyazda ve renk körlüğünde de okunur.")

    st.markdown("#### Kural Motoru Bulguları")
    if not m["flags"]:
        st.success("Kural motoru majör kırmızı bayrak yakalamadı. Bu 'temiz' "
                   "garantisi değildir — dipnot düzeyi burada görünmez; Claude "
                   "katmanına danış.")
    for sev, txt in m["flags"]:
        (st.error if sev == "yüksek" else st.warning)(f"**[{sev.upper()}]** {txt}")

    sn = m.get("sanity") or []
    with st.expander(f"🔍 Veri Tutarlılık Denetimi ({len(sn)} not)",
                     expanded=False):
        st.caption("Sessiz denetçi: kritik rakamlar iki bağımsız yoldan "
                   "hesaplanıp karşılaştırıldı. Buradaki notlar şirket "
                   "bayrağı değil, VERİ KALİTESİ uyarısıdır — bir metrik "
                   "burada geçiyorsa ona temkinli yaklaş.")
        ed = m.get("edgar") or {}
        kar = ed.get("karsilastirma") or []
        if kar:
            st.caption(f"🏛️ **SEC EDGAR çaprazı** — {ed.get('ozet', '')} · "
                       f"✓ ≤%2 (resmî teyit) · ≈ %2-10 (dönem farkı "
                       f"olabilir) · ✗ >%10 (SEC esastır)")
            st.dataframe(pd.DataFrame(
                [{"Kalem": p["kalem"],
                  "Yahoo": f"{p['yahoo']:,.0f}",
                  "SEC (10-K)": f"{p['sec']:,.0f}",
                  "SEC Yılı": p["sec_yil"],
                  "Sapma %": p["sapma_pct"],
                  "Uyum": p["uyum"]} for p in kar]),
                hide_index=True, width="stretch")
        elif ed.get("durum") not in ("OK", "KAPALI", None):
            st.caption("🏛️ SEC EDGAR'a ulaşılamadı (ağ/limit) — Yahoo tek "
                       "kaynak olarak kullanıldı.")
        if not sn:
            st.success("Tüm çapraz kontroller tutarlı — veri temiz "
                       "görünüyor.")
        for sev, txt in sn:
            (st.warning if sev == "uyarı" else st.info)(txt)

    mu = m.get("mutabakat")
    if mu:
        st.markdown("#### 🔍 SEC Mutabakatı")
        st.dataframe(pd.DataFrame([
            {"Kalem": s["kalem"], "Yıl": s["yil"],
             "Hesaplanan": f"{s['hesaplanan']:,.0f}",
             "SEC Raporlanan": f"{s['sec_raporlanan']:,.0f}",
             "Sapma": f"%{s['sapma']*100:.2f}",
             "Durum": ("✔ tutuyor" if s["durum"] == "OK"
                       else "✖ SAPMA")}
            for s in mu["satirlar"]]),
            hide_index=True, width="stretch")
        (st.warning if mu["sapan"] else st.success)("🔍 " + mu["not"])


def render_flow_tab(m, edu=False):
    if edu:
        st.caption("📚 **Nasıl okunur:** Bu sekme 'yakın gelecek + para "
                   "akışı' katmanıdır. *Katalizör* fiyatı oynatabilecek "
                   "takvimli olay; *revizyon momentumu* analist rüzgârının "
                   "yönü; *içeriden işlemler* şirketi en iyi bilenlerin kendi "
                   "parasıyla ne yaptığıdır. Üçü de tek başına al/sat sinyali "
                   "değil, tezin yönünü sınayan kanıttır.")

    # ── İMA EDİLEN HAREKET (kazanç beklentisi) ───────────────────────
    im = m.get("implied_move")
    if im and im.get("durum") == "OK":
        st.markdown("#### 🎯 Kazanç Beklentisi — İma Edilen Hareket")
        im1, im2, im3 = st.columns(3)
        im1.metric("Beklenen Mutlak Hareket",
                   f"±%{im['beklenen_hareket_pct']*100:.1f}",
                   (f"1σ: ±%{im['bir_sigma_pct']*100:.1f}"
                    if im.get("bir_sigma_pct") else None), delta_color="off",
                   help="Kazançtan sonraki ilk vadenin ATM straddle'ından "
                        "(×0.85) — piyasanın kote ettiği rakam budur. "
                        "DİKKAT: bu ~1σ DEĞİL, ~%50 kapsayan BEKLENEN "
                        "MUTLAK harekettir. Gerçek 1σ (~%68) alt satırda "
                        "ve daha geniştir.")
        im2.metric("Beklenen Bant",
                   f"{m['cur']}{im['alt_bant']:,.2f}–{im['ust_bant']:,.2f}",
                   help=f"Vade: {im['vade']} ({im.get('vade_gun','—')} gün)")
        im3.metric("ATM İma Edilen Volatilite",
                   f"%{im['atm_iv']*100:.0f}" if im.get("atm_iv") else "—",
                   help="Kazanç sonrası bu volatilite ~%90 olasılıkla söner "
                        "(IV crush).")
        st.caption("🎯 " + im["not"])
    elif im and im.get("durum") not in (None, "OK"):
        st.caption("🎯 İma edilen hareket: " + im.get("not", "hesaplanamadı"))

    # ── SEC 8-K MATERYAL OLAYLAR ─────────────────────────────────────
    o8 = m.get("olaylar_8k")
    if o8 and o8.get("durum") == "OK":
        kb = o8.get("kirmizi_bayrak")
        if kb:
            st.markdown("#### 🚩 SEC 8-K — Kırmızı Bayrak Olayları")
            for o in kb:
                its = " · ".join(f"**{it['kod']}** {it['aciklama']}"
                                 for it in o["itemler"])
                st.error(f"⚠️ {o['tarih']} — {its}  ·  "
                         f"[8-K metni]({o['link']})")
        with st.expander("📄 Son 8-K bildirimleri (tümü)"):
            for o in (o8.get("olaylar") or [])[:12]:
                sev = o["en_yuksek_onem"]
                ikon = {"kritik": "🔴", "yuksek": "🟠", "orta": "🟡",
                        "dusuk": "⚪"}.get(sev, "⚪")
                its = ", ".join(f"{it['kod']} {it['aciklama']}"
                                for it in o["itemler"])
                st.markdown(f"{ikon} **{o['tarih']}** — {its} · "
                            f"[link]({o['link']})")
        st.caption("🚩 " + o8.get("not", ""))

    st.markdown("#### 📅 Katalizör Takvimi")
    cat = m.get("catalysts") or {}
    gel = cat.get("gelecek") or []
    if gel:
        st.dataframe(pd.DataFrame(gel).rename(
            columns={"olay": "Olay", "tarih": "Tarih",
                     "gun_kaldi": "Gün Kaldı"}),
            hide_index=True, width="stretch")
    else:
        st.info("Yahoo takviminde yaklaşan kayıtlı olay yok. Şirkete özel "
                "olayları (sözleşme, regülasyon, dava) Claude raporu webden "
                "arar.")
    sur = cat.get("kazanc_surpriz_gecmisi") or []
    if sur:
        st.caption("Kazanç sürpriz geçmişi (son 4 çeyrek) — tahmini "
                   "tutturuyor mu?")
        st.dataframe(pd.DataFrame(sur).rename(
            columns={"tarih": "Tarih", "tahmin": "Tahmin EPS",
                     "gerceklesen": "Gerçekleşen", "surpriz": "Sürpriz %"}),
            hide_index=True, width="stretch")

    pdz = m.get("pead")
    if pdz:
        st.markdown("#### 📈 PEAD — Kazanç Sürprizi Sürüklenmesi")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Son Sürpriz", f"%{pdz['son_surpriz_pct']:+.1f}",
                  pdz.get("yon"), delta_color="off",
                  help="Bernard-Thomas (1989): sürpriz sonrası fiyat "
                       "haftalar-aylar aynı yönde sürüklenme "
                       "eğilimindedir — 3 aylık ufka uyan kanıtlı sinyal.")
        q2.metric("SUE (~)",
                  (f"{pdz['sue_yaklasik']:+.1f}"
                   if pdz.get("sue_yaklasik") is not None else "—"),
                  help="Son sürpriz ÷ son sürprizlerin std'si. |SUE| "
                       "büyüdükçe sürpriz o şirket için 'alışılmadık' "
                       "demektir.")
        q3.metric("Kaç Gün Önce",
                  (f"{pdz['gun_once']}g"
                   if pdz.get("gun_once") is not None else "—"),
                  ("sürüklenme penceresinde (~90g)"
                   if pdz.get("suruklenme_penceresinde")
                   else "pencere dışı"), delta_color="off")
        q4.metric("Ardışık Aynı Yön", pdz.get("ardisik_ayni_yon", "—"),
                  help="Üst üste kaç çeyrek aynı yönde sürpriz — sürpriz "
                       "momentumu da kalıcılık gösterir.")
        st.caption("📈 " + (pdz.get("not") or ""))

    # ── HABER AKIŞI (tam metin, kişisel kullanım) ────────────────────
    hb = m.get("haberler")
    if hb and hb.get("durum") == "OK" and hb.get("haberler"):
        st.markdown("#### 📰 Son Haberler")
        for h in hb["haberler"][:10]:
            ust = h["baslik"]
            alt_parts = [x for x in (h.get("kaynak"), h.get("tarih"),
                                     (f"{h['gun_once']}g önce"
                                      if h.get("gun_once") is not None
                                      else None)) if x]
            with st.expander(f"📰 {ust}"):
                if alt_parts:
                    st.caption(" · ".join(alt_parts))
                if h.get("ozet"):
                    st.write(h["ozet"])
                if h.get("link"):
                    st.markdown(f"[Kaynağı aç →]({h['link']})")
        st.caption("📰 " + hb.get("not", ""))
    elif hb and hb.get("durum") == "BOŞ":
        st.caption("📰 Bu hisse için güncel haber akışı bulunamadı.")

    st.markdown("#### 📈 Analist Revizyon Momentumu")
    rv = m.get("revizyon") or {}
    if rv:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Yön", rv.get("yon", "—"), help=GLOSSARY["revizyon"])
        r2.metric("Net Revizyon (30g)",
                  rv.get("skor_net_30g", "—"),
                  help="Yukarı revize eden analist − aşağı revize eden. "
                       "Pozitif = rüzgâr arkadan.")
        e0 = rv.get("eps_bu_yil") or {}
        r3.metric("Bu Yıl EPS (90g Δ)", fmt_pct(e0.get("degisim")),
                  help="Bu yılın konsensüs EPS tahmini 90 gün öncesine göre "
                       "ne kadar değişti.")
        pt = rv.get("hedef_fiyat") or {}
        r4.metric("Hedef Fiyat Potansiyeli", fmt_pct(pt.get("potansiyel")),
                  help=f"Analist ortalama hedefi "
                       f"({pt.get('ortalama', '—')}) mevcut fiyata göre.")
        od = rv.get("oneri_dagilimi") or {}
        if od:
            st.caption("Öneri dağılımı: " + " · ".join(
                f"{k}: {v}" for k, v in od.items()))
    else:
        st.info("Analist tahmin verisi bulunamadı (küçük/az takip edilen "
                "şirketlerde normaldir).")

    ca_ = m.get("call")
    if ca_:
        st.markdown("#### 🎙️ Kazanç Çağrısı Dili")
        t1_, t2_, t3_, t4_ = st.columns(4)
        t1_.metric("Ton", ca_["ton"])
        t2_.metric("Hedge Kelime", ca_["hedge_sayisi"],
                   help="'challenging, headwinds, uncertain' türü — "
                        "yönetimin sakınma dili.")
        t3_.metric("Güven Kelime", ca_["guven_sayisi"],
                   help="'record, strong, raise' türü.")
        t4_.metric("Rehber Geçişi", ca_["rehber_gecisi"])
        st.caption(ca_.get("not", "") + " Metin Claude raporuna aktarıldı.")

    st.markdown("#### 👤 İçeriden İşlemler (Form 4)")
    ins = m.get("insider") or {}
    oz = ins.get("ozet") or {}
    if oz:
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Alım (12 ay)", oz.get("alim_adet", 0),
                  help=GLOSSARY["insider_g"])
        i2.metric("Satım (12 ay)", oz.get("satim_adet", 0))
        a_usd, s_usd = oz.get("alim_usd"), oz.get("satim_usd")
        i3.metric("Alım $", f"${a_usd/1e6:,.1f}M" if a_usd else "—")
        i4.metric("Satım $", f"${s_usd/1e6:,.1f}M" if s_usd else "—")
        _kayn = oz.get("kaynak") or "Yahoo"
        _pl = oz.get("planli_satis_adet")
        _lnk = ins.get("sec_link") or m.get("insider_link")
        st.caption("Kaynak: **" + _kayn + "**"
                   + (f" · 10b5-1 planlı satış: {_pl}"
                      if _pl is not None else "")
                   + ((" · [SEC Form 4 kayıtları](" + _lnk + ")")
                      if _lnk else ""))
        if not oz.get("alim_adet"):
            st.caption("ℹ️ Açık piyasa ALIMI görünmüyor — bu çoğu şirkette "
                       "NORMALDİR (işlemlerin büyük kısmı satış / opsiyon "
                       "kullanımı / ödüldür); yokluk tek başına ayı sinyali "
                       "DEĞİLDİR. Küme alımı (30 günde ≥3 yönetici) "
                       "görülürse tablo zaten yeşil bayrak basar.")
        if oz.get("kume_alimi_30g_3insider"):
            st.success("🟢 KÜME ALIMI: 30 gün içinde ≥3 farklı yönetici alım "
                       "yapmış — akademik olarak en güçlü içeriden "
                       "sinyallerden biri.")
        son = ins.get("son_islemler") or []
        if son:
            st.dataframe(pd.DataFrame(son).rename(
                columns={"tarih": "Tarih", "kisi": "Kişi", "unvan": "Unvan",
                         "tur": "Tür", "adet": "Adet",
                         "deger_usd": "Değer $", "aciklama": "Açıklama"}),
                hide_index=True, width="stretch")
        st.caption(ins.get("not", ""))
    else:
        st.info("İçeriden işlem verisi alınamadı: SEC Form 4 erişimi ve "
                "Yahoo özeti boş döndü. Bu 'işlem yok' demek DEĞİLDİR — "
                "'veri gelmedi' demektir; aşağıdaki linkten doğrudan "
                "kontrol edebilirsin.")
        if m.get("insider_link"):
            st.markdown("[SEC Form 4 kayıtlarını doğrudan aç]("
                        + m["insider_link"] + ")")

    st.markdown("#### 🩳 Açığa Satış (Short Interest)")
    sh = m.get("short") or {}
    if sh:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Float'ın %'si", fmt_pct(sh.get("float_yuzdesi")),
                   help=GLOSSARY["short"])
        sc2.metric("Kapatma Günü", fmt_f(sh.get("kapatma_gunu")),
                   help="Tüm kısa pozisyonun ortalama hacimle kaç günde "
                        "kapanacağı. 5+ gün, sıkışma potansiyeli demektir.")
        sc3.metric("Aylık Değişim", fmt_pct(sh.get("aylik_degisim")))
        sc4.metric("Veri Tarihi", sh.get("veri_tarihi") or "—")
        st.caption(sh.get("not", ""))
    else:
        st.info("Açığa satış verisi yok.")

    st.markdown("#### 🏦 Kurumsal Sahiplik (13F)")
    ku = m.get("kurumsal") or {}
    if ku:
        u1, u2, u3 = st.columns(3)
        u1.metric("Kurum Payı", fmt_pct(ku.get("kurum_yuzde")),
                  help=GLOSSARY["kurumsal"])
        u2.metric("İçeriden Payı", fmt_pct(ku.get("insider_yuzde")))
        ks = ku.get("kurum_sayisi")
        u3.metric("Kurum Sayısı", f"{ks:,.0f}" if ks else "—")
        top5 = ku.get("en_buyuk_5") or []
        if top5:
            st.dataframe(pd.DataFrame(
                [{"Kurum": t["kurum"],
                  "Pay %": fmt_pct(t.get("pay_yuzde")),
                  "Çeyreklik Δ": fmt_pct(t.get("ceyreklik_degisim"))}
                 for t in top5]), hide_index=True, width="stretch")
        st.caption(ku.get("not", ""))
    else:
        st.info("Kurumsal sahiplik verisi yok.")

    ipo = m.get("ipo") or {}
    if ipo.get("durum") in ("LOCKUP_GELECEK", "LOCKUP_DOLDU"):
        st.markdown("#### 🔓 IPO Lock-up")
        if ipo["durum"] == "LOCKUP_GELECEK":
            st.warning(f"Genç halka arz (IPO {ipo['ipo_tarihi']}). Tahminî "
                       f"lock-up bitişi **{ipo['lockup_tahmini']}** "
                       f"(~{ipo['kalan_gun']} gün) — kilit açılınca insider "
                       f"arzı fiyat baskısı yaratabilir. {ipo.get('not', '')}")
        else:
            st.info(f"IPO {ipo['ipo_tarihi']}; tahminî lock-up "
                    f"{ipo['lockup_tahmini']} itibarıyla DOLDU — insider "
                    f"satışları serbest. {ipo.get('not', '')}")


def render_claude_tab(m, api_key, model, use_web):
    st.caption("Bu katman, sayısal motorun çıktısını Claude'a taşır: hüküm, "
               "ters-DCF sorgusu, adli notlar, boğa/ayı ve modelin kendi "
               "eleştirisi. API anahtarınla, senin makinenden çağrılır.")
    ready = bool(api_key)
    if not ready:
        st.warning("Claude raporu için kenar çubuğu → **Claude Ayarları** "
                   "bölümünden API anahtarı gir (veya ANTHROPIC_API_KEY "
                   "ortam değişkenini ayarla). **API'ye para yüklemek "
                   "istemiyorsan aşağıdaki Manuel Mod'u kullan — aynı "
                   "analizi ek ücret ödemeden alırsın.**")

    with st.expander("🆓 Manuel Mod — API'siz kullanım (ek ücret yok)",
                     expanded=not ready):
        st.caption(
            "Claude aboneliği (Pro/Max) ile Claude API AYRI ürünlerdir; "
            "abonelik bu programa API erişimi vermez — bu Anthropic'in "
            "faturalandırma yapısıdır, kodla aşılamaz. Çözüm: program "
            "istemi hazırlar, sen claude.ai sohbetine yapıştırırsın. "
            "Analiz aynı analizdir; tek kayıp otomasyondur (rapor sohbette "
            "kalır, uygulamaya geri yazılmaz).")
        mt1, mt2 = st.columns([3, 2])
        m_tur = mt1.radio("Hangi istem?",
                          ["🧠 Derin Analiz (IC memo)",
                           "🧩 5 Soru — Şirketi Anlat"],
                          horizontal=False)
        m_kisa = mt2.toggle("Paketi kısalt", value=True,
                            help="Uzun sohbet sınırına takılmamak için "
                                 "hacimli bölümleri çıkarır; çıkarılanların "
                                 "listesi istemin içinde yazılıdır.")
        tur = "5soru" if m_tur.startswith("🧩") else "derin"
        try:
            istem = manuel_istem(m, tur, m_kisa)
        except Exception as e:
            istem = None
            st.error(f"İstem üretilemedi: {e}")
        if istem:
            n = len(istem)
            st.caption(f"İstem uzunluğu: **{n:,} karakter** · "
                       + ("Sohbete doğrudan yapıştırabilirsin."
                          if n < 45000 else
                          "Uzun — aşağıdan .md indirip sohbete DOSYA olarak "
                          "eklemen daha güvenli."))
            st.download_button(
                "📄 İstemi indir (.md) — sohbete dosya olarak ekle",
                istem.encode("utf-8"),
                file_name=f"{m['ticker']}_{tur}_istem.md",
                mime="text/markdown", key=f"dl_manuel_{tur}")
            with st.expander("İstemi ekranda göster (kopyala)"):
                st.code(istem, language="markdown")
            st.info("Kullanım: (1) yukarıdaki dosyayı indir ya da metni "
                    "kopyala · (2) claude.ai sohbetinde yeni bir konuşma "
                    "aç · (3) dosyayı ekle veya metni yapıştır, gönder · "
                    "(4) rapor sohbette üretilir. Web araması istiyorsan "
                    "sohbette arama özelliğinin açık olduğundan emin ol.")
    cbt1, cbt2 = st.columns([2, 3])
    kirmizi = cbt2.toggle(
        "🔴 Kırmızı Takım turu", value=False,
        help="İkinci bir çağrıyla ilk rapor ÇÜRÜTÜLMEYE çalışılır: en "
             "güçlü karşı-tez, paketle çelişen cümleler, aşırı güven. "
             "Ayakta kalan tez güçlü tezdir. Ek maliyet: ~1 çağrı.")
    if cbt1.button("🧠 Derin Analiz Raporu Üret", type="primary",
                   disabled=not ready):
        payload = build_payload(m)
        _nit = st.session_state.get("claude_nitel")
        if _nit:
            # 5 Soru cevapları karar kartına akar (talimat 20): anlaşılırlık
            # notu düşükse model hükmü otomatik temkinli çerçeveler.
            payload["nitel_5_soru_cevaplari"] = _nit[:4000]
        with st.spinner(f"Claude ({model}) dosyayı inceliyor…"):
            try:
                report = call_claude(api_key, model, payload, use_web)
                st.session_state["claude_report"] = report
                st.session_state["claude_meta"] = (
                    f"{m['ticker']} · {model} · "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
                st.session_state.pop("claude_redteam", None)
            except Exception as e:
                report = None
                st.error(f"Claude çağrısı başarısız: {e}")
                with st.expander("Teknik ayrıntı"):
                    st.code(traceback.format_exc())
        if kirmizi and report:
            with st.spinner("🔴 Kırmızı takım raporu çürütmeye çalışıyor…"):
                try:
                    st.session_state["claude_redteam"] = call_claude_redteam(
                        api_key, model, payload, report)
                except Exception as e:
                    st.session_state["claude_redteam"] = (
                        f"_Kırmızı takım çağrısı başarısız: {e}_")

    st.markdown("---")
    st.markdown("##### 🧩 5 Soru — Şirketi Anlat (nitel katman)")
    st.caption("Sayıların cevaplayamadığı, 10-K metnine ait beş temel "
               "soruyu Claude araştırıp SADE Türkçe cevaplar: ne satıyor, "
               "neden bundan alıyorlar, onu ne öldürür, yönetim sözünü "
               "tuttu mu, kâr nereden geliyor. Doğrulayamadığı yere "
               "'BİLİNMİYOR' yazar — boşluk uydurulmaz.")
    if st.button("🧩 5 Soruyu Cevapla", disabled=not ready):
        with st.spinner("Claude şirketi anlatmaya hazırlanıyor…"):
            try:
                st.session_state["claude_nitel"] = call_claude_nitel(
                    api_key, model, build_payload(m), use_web)
                st.session_state["claude_nitel_meta"] = (
                    f"{m['ticker']} · {model} · "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                st.error(f"5 Soru çağrısı başarısız: {e}")
    nitel = st.session_state.get("claude_nitel")
    if nitel:
        st.caption(st.session_state.get("claude_nitel_meta", ""))
        st.markdown(nitel)
        st.info("🧭 Bu bölüm senin yerine METNİ okur, KARARI vermez. "
                "Cevaplar ikna edici değilse doğru hamle 'daha çok "
                "araştırmak' değil, o hisseyi PAS GEÇMEKTİR — küçük "
                "yatırımcının en güçlü kozu budur.")
        st.download_button(
            "5 Soru cevaplarını indir (.md)", nitel.encode("utf-8"),
            file_name=f"{m['ticker']}_5soru.md", mime="text/markdown",
            key="dl_nitel")

    report = st.session_state.get("claude_report")
    if report:
        st.caption(st.session_state.get("claude_meta", ""))
        st.markdown(report)
        rt = st.session_state.get("claude_redteam")
        if rt:
            st.markdown("---")
            st.markdown("### 🔴 Kırmızı Takım Karşı-Raporu")
            st.markdown(rt)
        indirilecek = report + (("\n\n---\n\n# KIRMIZI TAKIM "
                                 "KARŞI-RAPORU\n\n" + rt) if rt else "")
        if nitel:
            indirilecek += ("\n\n---\n\n# 5 SORU — ŞİRKETİ ANLAT\n\n"
                            + nitel)
        st.download_button(
            "Raporu indir (.md)", indirilecek.encode("utf-8"),
            file_name=f"{m['ticker']}_claude_analiz.md",
            mime="text/markdown")

    with st.expander("Claude'a giden veri paketini gör (JSON)"):
        st.json(build_payload(m))


# ══ TARAMA (SCREENER) KATMANI ═══════════════════════════════════════════
# İki kademeli mimari: (1) hafif veriyle hızlı ön eleme → Top-20 aday,
# (2) adayı tek tıkla mevcut DERİN analize gönder. Tarama aday BULUR,
# derin analiz HÜKÜM verir — birbirinin yerine geçmez.
# Evren listeleri GÖMÜLÜDÜR (~2025 bileşimi, yaklaşık): endeks bileşimleri
# değişir; veri gelmeyen/emekli semboller sessizce atlanır ve sayısı
# raporlanır. Kesinlik istersen 'Özel liste' kullan.




# ═══════════════════════════════════════════════════════════════════════
# ÇEKİLME GEÇİŞ MOTORU — "buradan daha ne kadar düşer?"
# ═══════════════════════════════════════════════════════════════════════
# Yöntem: ÜST ÜSTE BİNEN pencereler değil EPİZOT sayımı. Her epizot bir
# bağımsız gözlemdir; N gerçekten N'dir. Doğrulama: araştırmadaki
# Schwab/Morningstar rakamı (1974'ten beri 27 düzeltmenin 6'sı ayı = %22)
# aynı mantıkla üretilebiliyor.
# ═══════════════════════════════════════════════════════════════════════

ESIKLER = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MIN_EPIZOT = 10          # bunun altında "anlamlı değil" damgası


def cekilme_epizotlari(kapanis, min_derinlik=0.05, tanim="tam"):
    """Zirveden zirveye epizotlar — her biri BİR bağımsız gözlem.

    ⚠️ TANIM ÖNEMLİ ve yayınlanmış sayımlarla FARK YARATIR:

    tanim="tam" (varsayılan, MUHAFAZAKÂR):
        Epizot, fiyat ESKİ ZİRVEYE geri dönene kadar sürer. Yani %12
        düşüp kısmen toparlanıp sonra %20'ye inen bir hareket TEK
        epizottur (derinlik %20). Bu, "buradan daha ne kadar düşer"
        sorusu için DOĞRU tanımdır çünkü tek bir sürekli düşüşü ikiye
        bölmez.

    tanim="gevsek" (yayınlanmış sayımlara YAKIN):
        Dipten %`toparlanma_esigi` kadar yükseliş epizotu KAPATIR.
        Basın ve veri sağlayıcıları genelde böyle sayar — bu yüzden
        "1974'ten beri 27 düzeltme" gibi rakamlar çıkar. Daha ÇOK
        epizot verir ama epizotlar tam bağımsız değildir.

    Hangi tanımı kullandığın epizot SAYISINI belirgin değiştirir; çıktıda
    açıkça yazılır ki başka kaynaklarla karşılaştırırken şaşırmayasın.
    """
    c = pd.Series(kapanis).dropna()
    if len(c) < 250:
        return []
    zirve = c.cummax()
    dd = 1.0 - c / zirve

    epizotlar, i, n = [], 0, len(c)
    while i < n:
        if dd.iloc[i] <= 1e-9:          # zirvedeyiz
            i += 1
            continue
        bas = i                          # düşüş başladı
        z_deger = float(zirve.iloc[i])
        # bu zirveye geri dönene kadar ilerle
        j = i
        en_derin, dip_idx = 0.0, i
        while j < n and float(c.iloc[j]) < z_deger:
            d = float(dd.iloc[j])
            if d > en_derin:
                en_derin, dip_idx = d, j
            j += 1
        toparlandi = j < n
        if en_derin >= min_derinlik:
            epizotlar.append({
                "bas_idx": bas, "dip_idx": dip_idx,
                "bit_idx": (j if toparlandi else n - 1),
                "bas_tarih": _tarih(c, bas),
                "dip_tarih": _tarih(c, dip_idx),
                "bit_tarih": _tarih(c, j if toparlandi else n - 1),
                "derinlik": round(en_derin, 4),
                "dusus_gun": int(dip_idx - bas),
                "toparlanma_gun": (int(j - dip_idx) if toparlandi else None),
                "toparlandi": bool(toparlandi),
            })
        i = max(j, i + 1)

    if tanim == "gevsek":
        # ⚠️ ÖLÇÜLDÜ: gevşek tanım epizot sayısını 12 → 79 çıkardı ama
        # KOŞULLU ORAN neredeyse aynı kaldı (%40 → %42). Yani ek epizotlar
        # BAĞIMSIZ BİLGİ TAŞIMIYOR — aynı tarihsel düşüşleri parçalıyor.
        # N'i şişirmek güven aralıklarını yapay olarak DARALTIR ve
        # olduğundan emin görünmene yol açar. Bu yüzden varsayılan "tam".
        epizotlar = _gevsek_bol(c, dd, epizotlar, min_derinlik)
    for e in epizotlar:
        e["tanim"] = tanim
    return epizotlar


def _gevsek_bol(c, dd, epizotlar, min_derinlik, toparlanma_esigi=0.10):
    """Dipten %10 yükseliş epizotu kapatır — basın sayımına yakın."""
    out = []
    for e in epizotlar:
        seg = c.iloc[e["bas_idx"]:e["bit_idx"] + 1]
        if len(seg) < 3:
            out.append(e)
            continue
        alt, dip_i, dip_v = e["bas_idx"], e["dip_idx"], float(c.iloc[e["dip_idx"]])
        i = e["bas_idx"]
        yerel_dip_i, yerel_dip = i, float(c.iloc[i])
        z = float(c.iloc[e["bas_idx"]])
        parca_bas = i
        while i <= e["bit_idx"]:
            v = float(c.iloc[i])
            if v < yerel_dip:
                yerel_dip, yerel_dip_i = v, i
            # dipten %10 toparlandıysa bu parçayı kapat
            if yerel_dip > 0 and v / yerel_dip - 1 >= toparlanma_esigi:
                zz = float(c.iloc[parca_bas:i + 1].max())   # D2: koşan zirve
                d = 1.0 - yerel_dip / max(zz, 1e-9)
                if d >= min_derinlik:
                    out.append({**e, "bas_idx": parca_bas,
                                "dip_idx": yerel_dip_i, "bit_idx": i,
                                "bas_tarih": _tarih(c, parca_bas),
                                "dip_tarih": _tarih(c, yerel_dip_i),
                                "bit_tarih": _tarih(c, i),
                                "derinlik": round(d, 4),
                                "dusus_gun": int(yerel_dip_i - parca_bas),
                                "toparlanma_gun": int(i - yerel_dip_i),
                                "toparlandi": True})
                parca_bas = i
                yerel_dip, yerel_dip_i = v, i
            i += 1
        if parca_bas < e["bit_idx"]:
            seg2 = c.iloc[parca_bas:e["bit_idx"] + 1]
            zz = float(seg2.max())          # D2: parça zirvesi
            dp = float(seg2.min())
            d = 1.0 - dp / max(zz, 1e-9)
            if d >= min_derinlik:
                _di = parca_bas + int(seg2.values.argmin())
                out.append({**e, "bas_idx": parca_bas,
                            "dip_idx": _di, "bit_idx": e["bit_idx"],
                            "bas_tarih": _tarih(c, parca_bas),
                            "dip_tarih": _tarih(c, _di),
                            "bit_tarih": _tarih(c, e["bit_idx"]),
                            "derinlik": round(d, 4),
                            "dusus_gun": int(_di - parca_bas),
                            "toparlanma_gun": int(e["bit_idx"] - _di),
                            "toparlandi": e.get("toparlandi", False)})
    return out or epizotlar


def _tarih(c, i):
    try:
        t = c.index[i]
        return str(pd.Timestamp(t).date())
    except Exception:
        return None


def gecis_tablosu(epizotlar, esikler=None):
    """P(derinliğe ulaşır | şu eşiğe ulaştı) — bağımsız N ile."""
    es = esikler or ESIKLER
    if not epizotlar:
        return None
    der = np.array([e["derinlik"] for e in epizotlar])
    tablo = []
    for i, kosul in enumerate(es):
        ulasan = der >= kosul - 1e-9
        n = int(ulasan.sum())
        if n == 0:
            continue
        satir = {"kosul": kosul, "n_epizot": n, "gecisler": {}}
        for hedef in es[i + 1:]:
            k = int((der[ulasan] >= hedef - 1e-9).sum())
            satir["gecisler"][hedef] = {
                "oran": round(k / n, 3), "sayi": k, "toplam": n,
                # binom %95 GA (Wilson) — küçük N'de normal yaklaşım yalan söyler
                "ga95": _wilson(k, n),
            }
        satir["yeterli"] = n >= MIN_EPIZOT
        tablo.append(satir)
    return tablo


def _wilson(k, n, z=1.96):
    """Wilson güven aralığı — küçük örneklemde doğru olan."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    yc = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, m - yc), 3), round(min(1.0, m + yc), 3))


def bugun_nerede(kapanis, epizotlar, esikler=None):
    """Şu anki çekilme + buradan tarihsel olarak nereye gidilmiş."""
    c = pd.Series(kapanis).dropna()
    if len(c) < 250 or not epizotlar:
        return None
    es = esikler or ESIKLER
    zirve = float(c.cummax().iloc[-1])
    son = float(c.iloc[-1])
    simdi = 1.0 - son / zirve

    # bugünkü çekilmeyi AŞAN en yakın eşik = koşul
    kosul = None
    for e in es:
        if simdi >= e - 1e-9:
            kosul = e
    der = np.array([x["derinlik"] for x in epizotlar])

    if kosul is None:
        return {
            "cekilme": round(simdi, 4), "zirve": round(zirve, 2),
            "son": round(son, 2), "kosul": None,
            "mesaj": (
                f"Zirveden %{simdi*100:.1f} aşağıdasın — bu, epizot "
                f"sayacının en küçük eşiğinin (%{es[0]*100:.0f}) bile "
                f"altında. Tarihsel olarak bu kadar sığ hareketler "
                f"OLAĞANDIR ve ayrı bir 'olay' sayılmaz. Buradan "
                f"çıkarılacak bir taban oran YOK — çünkü henüz bir şey "
                f"olmuş değil."),
            "satir": None,
        }

    ulasan = der >= kosul - 1e-9
    n = int(ulasan.sum())
    gecis = []
    for hedef in es:
        if hedef <= kosul + 1e-9:
            continue
        k = int((der[ulasan] >= hedef - 1e-9).sum())
        lo, hi = _wilson(k, n)
        gecis.append({"hedef": hedef, "oran": round(k / n, 3),
                      "sayi": k, "toplam": n, "ga95": (lo, hi),
                      "fiyat": round(zirve * (1 - hedef), 2)})
    return {
        "cekilme": round(simdi, 4), "zirve": round(zirve, 2),
        "son": round(son, 2), "kosul": kosul, "n_epizot": n,
        "yeterli": n >= MIN_EPIZOT, "gecis": gecis,
        "toplam_epizot": len(epizotlar),
    }


def duz_turkce(bn, endeks_adi="Endeks"):
    """İstatistiği DÜZ CÜMLEYE çevir — asıl iş bu."""
    if not bn:
        return None
    if bn.get("kosul") is None:
        return {"basli": "Henüz bir 'olay' yok", "govde": bn["mesaj"],
                "satirlar": [], "uyari": None}

    n = bn["n_epizot"]
    basli = (f"{endeks_adi} zirveden %{bn['cekilme']*100:.1f} aşağıda "
             f"(%{bn['kosul']*100:.0f} eşiğini geçmiş)")
    govde = (
        f"Tarihte bu endeks {bn['toplam_epizot']} kez zirveden en az "
        f"%5 düştü. Bunlardan **{n} tanesi** en az %{bn['kosul']*100:.0f} "
        f"düştü — yani bugün bulunduğun yere geldi. O {n} olayda "
        f"buradan sonra ne olduğu şöyle:")
    tanim_notu = (
        "📐 **Olay nasıl sayıldı:** Bir 'olay', fiyatın zirveden düşüp "
        "O ZİRVEYE GERİ DÖNMESİNE kadar süren dönem. Yani %12 düşüp "
        "kısmen toparlanıp sonra %20'ye inen bir hareket TEK olaydır. "
        "Basında ve veri sağlayıcılarda genelde daha gevşek sayılır "
        "(dipten %10 yükseliş olayı kapatır), o yüzden onların "
        "rakamları DAHA ÇOK olay gösterir. Ölçtüm: gevşek sayım olay "
        "sayısını ~6 kat artırıyor ama koşullu oranı neredeyse hiç "
        "değiştirmiyor — yani fazladan olaylar bağımsız bilgi "
        "taşımıyor, sadece N'i şişirip seni olduğundan emin gösteriyor. "
        "Bu yüzden muhafazakâr sayım kullanıldı.")

    satirlar = []
    for g in bn["gecis"]:
        lo, hi = g["ga95"]
        satirlar.append({
            "hedef_yuzde": g["hedef"],
            "hedef_fiyat": g["fiyat"],
            "cumle": (
                f"**%{g['hedef']*100:.0f}'e kadar indi:** {g['sayi']} / "
                f"{g['toplam']} olayda  (yani ~%{g['oran']*100:.0f})"),
            "ga_cumle": (f"gerçek oran %95 güvenle "
                         f"%{lo*100:.0f}–%{hi*100:.0f} arasında"
                         if lo is not None else ""),
            "fiyat_cumle": (f"bu, {endeks_adi} {g['fiyat']:,.0f} "
                            f"seviyesine denk gelir"),
            "oran": g["oran"],
        })

    uyari = None
    if not bn["yeterli"]:
        uyari = (
            f"⚠️ **BU SAYILARA GÜVENME.** Yalnız {n} bağımsız olay var "
            f"({MIN_EPIZOT} altı). Bir oranı {n} gözlemden çıkarmak, "
            f"{n} kez yazı-tura atıp 'bu para hileli' demeye benzer. "
            f"Yukarıdaki güven aralıklarına bak — muhtemelen o kadar "
            f"geniştir ki hiçbir şey söylemiyorlar.")
    elif n < 20:
        uyari = (
            f"ℹ️ {n} bağımsız olay — okunabilir ama ince. Güven "
            f"aralıkları geniş; oranı 'yaklaşık' oku, 'kesin' değil.")
    return {"basli": basli, "govde": govde, "satirlar": satirlar,
            "uyari": uyari, "tanim_notu": tanim_notu}

# ═══════════════════════════════════════════════════════════════════════
# PİYASA BAĞLAMI — Nasdaq / S&P endeks rejimi
# ═══════════════════════════════════════════════════════════════════════
# DOKTRİN: Endeks seviyesi "NE ZAMAN al" demez, "NE KADAR al" der.
# Gerekçe ve reddedilen özellikler modülün içinde yazılı.
# ═══════════════════════════════════════════════════════════════════════

# ── Endeksler ────────────────────────────────────────────────────────
ENDEKSLER = {
    "^IXIC": "Nasdaq Bileşik",
    "^NDX": "Nasdaq-100",
    "^GSPC": "S&P 500",
}
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"

# ── TABAN ORANLAR (araştırmadan; ETİKETLİ ve YAKLAŞIK) ──────────────
# Kaynak: T. Rowe Price (S&P 500, Oca 1940–Ara 2025) ve benzeri çalışmalar.
# DİKKAT: bu gözlemler ÖRTÜŞEN pencerelerden gelir ve birkaç tarihsel
# episoda hâkimdir — göründüğünden istatistiksel olarak ZAYIFTIR.
CEKILME_TABAN = [
    (0.00, 0.05, "zirveye yakın",
     "Tarihsel 12 aylık ortalama getiri koşulsuz ortalamaya YAKIN "
     "(~%12-13). Zirveye yakın olmak kötü bir alım zamanı DEĞİLDİR — "
     "endeksler zamanın çoğunu zirveye yakın geçirir."),
    (0.05, 0.10, "hafif çekilme",
     "Yılların ~2/3'ünde en az %10 çekilme görülür; %5-10 aralığı "
     "OLAĞANDIR. Forward getiri koşulsuzdan belirgin farklı değil."),
    (0.10, 0.20, "düzeltme bölgesi",
     "Tarihsel olarak forward getiri koşulsuz ortalamanın biraz "
     "ÜSTÜNDE, ama dağılım geniş: %10 çekilmelerin azımsanmayacak "
     "kısmı %20+'ye derinleşir."),
    (0.20, 1.01, "ayı bölgesi",
     "T. Rowe Price (1940-2025): %20 çekilme sonrası ortalama 12 aylık "
     "getiri ~%17.3, koşulsuz ~%12.8. AMA en kötü senaryolar hâlâ "
     "negatif — bu bir GARANTİ değil, dağılım kayması."),
]

VIX_TABAN = [
    (0.0, 0.25, "düşük oynaklık",
     "Sakin rejim. Tarihsel olarak forward getiri ortalamaya yakın; "
     "ama düşük VIX rehavet demektir, sürpriz burada acıtır."),
    (0.25, 0.75, "normal", "Olağan bant — ayırt edici bilgi taşımaz."),
    (0.75, 0.90, "yüksek", "Gerginlik artmış; pozisyon boyutunu gözden geçir."),
    (0.90, 1.01, "aşırı yüksek",
     "VIX ~80. yüzdelik üstünde: literatürde forward 6-12 aylık getiri "
     "ORTALAMANIN ÜSTÜNDE (Giot 2005 ve sonrası; R² 6ay ~%20, 12ay "
     "~%30 bildirilmiş). Kısa ufuktaki kanıtı en güçlü teknik değişken "
     "— ama bu bir RİSK PRİMİ hasadıdır, kesinlik değil."),
]

# ── Yoğunlaşma referansı (2025 sonu; YILDA BİR GÜNCELLE) ────────────
YOGUNLASMA = {
    "^GSPC": {"ilk10": 0.407, "tarih": "2025-12",
              "not": "S&P 500 ilk 10 ağırlığı 2015 sonunda ~%19'du."},
    "^NDX": {"ilk10": 0.50, "tarih": "2025-12",
             "not": "Nasdaq-100 daha yoğun; 'Mag 7' Temmuz 2023 özel "
                    "dengelemesi öncesi ~%55'e ulaşmıştı."},
}

# ── Faiz duyarlılığı (Nasdaq kendi çalışması, 1985-2021) ────────────
FAIZ_DUYARLILIK = {
    "^NDX": 2.00, "^IXIC": 1.80, "^GSPC": 0.72,
    "_not": ("Nasdaq'ın kendi çalışması (441 ay): NDX 2.0 · SPX 0.72 · "
             "Dow 1.28 (yüzde puanlık getiri değişimi / %1 faiz hareketi). "
             "AMA bu ilişki REJİME BAĞLI ve İŞARET DEĞİŞTİRİR: düşük "
             "faiz seviyelerinden yükselişte hisseler ARTABİLİR. Sabit "
             "bir beta olarak KULLANILMAZ."),
}


def _yuzdelik(seri, deger):
    """Değerin kendi tarihindeki yüzdelik konumu."""
    try:
        s = pd.Series(seri).dropna()
        if len(s) < 60 or deger is None:
            return None
        return float((s < float(deger)).mean())
    except Exception:
        return None


def _kova(deger, tablo):
    for alt, ust, ad, aciklama in tablo:
        if alt <= deger < ust:
            return {"ad": ad, "aciklama": aciklama,
                    "bant": (alt, ust)}
    return None


def endeks_rejimi(kapanis, vix_seri=None, vix_son=None):
    """Endeksin ŞU ANKİ rejimi — tanımlayıcı, sinyal DEĞİL."""
    if kapanis is None or len(kapanis) < 220:
        return None
    c = pd.Series(kapanis).dropna()
    son = float(c.iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    zirve52 = float(c.iloc[-252:].max()) if len(c) >= 252 else float(c.max())
    cekilme = 1.0 - son / zirve52 if zirve52 else None

    out = {
        "son": round(son, 2),
        "ma200": round(ma200, 2),
        "ma200_ustu": bool(son >= ma200),
        "ma200_mesafe": round(son / ma200 - 1, 4),
        "zirve_52h": round(zirve52, 2),
        "cekilme": round(cekilme, 4) if cekilme is not None else None,
        "cekilme_kova": _kova(cekilme, CEKILME_TABAN) if cekilme is not None
                        else None,
        "trend_notu": (
            "Fiyat 200 günlük ortalamanın ÜSTÜNDE. Faber'in kuralı "
            "burada 'yatırımda kal' der — ama bu kuralın kanıtlanmış "
            "faydası GETİRİ ARTIRMAK değil, OYNAKLIK ve ÇEKİLME "
            "AZALTMAKTIR."
            if son >= ma200 else
            "Fiyat 200 günlük ortalamanın ALTINDA. Tarihsel olarak bu "
            "rejimde oynaklık ve çekilme riski yüksektir. Bu bir 'satış "
            "sinyali' DEĞİL, risk bütçesini kısma gerekçesidir."),
    }
    if vix_son is not None:
        vy = _yuzdelik(vix_seri, vix_son) if vix_seri is not None else None
        out["vix"] = round(float(vix_son), 2)
        out["vix_yuzdelik"] = round(vy, 3) if vy is not None else None
        out["vix_kova"] = _kova(vy, VIX_TABAN) if vy is not None else None
    return out


# O1: elle girilen endeks F/K-CAPE için gömülü YAKLAŞIK konum ızgarası
# (Shiller 1881–2024 beşliklerinden yuvarlanmış; kesin yüzdelik DEĞİL).
_ELLE_CAPE_IZGARA = [(5.0, 0.02), (9.0, 0.10), (11.7, 0.25),
                     (16.0, 0.50), (21.0, 0.75), (25.5, 0.85),
                     (28.0, 0.90), (31.0, 0.95), (44.0, 0.995)]


def _elle_deger_baglami(son):
    try:
        v = float(son)
    except Exception:
        return None
    g = _ELLE_CAPE_IZGARA
    if v <= g[0][0]:
        y = g[0][1]
    elif v >= g[-1][0]:
        y = g[-1][1]
    else:
        y = g[0][1]
        for (x0, y0), (x1, y1) in zip(g, g[1:]):
            if x0 <= v <= x1:
                y = y0 + (y1 - y0) * (v - x0) / (x1 - x0)
                break
    return {"olcut": "CAPE/F-K (elle) [YAKLAŞIK]",
            "deger": round(v, 2), "yuzdelik": round(y, 3),
            "okuma": (f"Girilen {v:.1f}, YAKLAŞIK Shiller (1881–2024) "
                      f"dağılımında ~%{y*100:.0f}. yüzdelik"),
            "kaynak": "gömülü yaklaşık ızgara — kaba konum",
            "ufuk_uyarisi": (
                "⚠️ ZAMANLAMA SİNYALİ DEĞİLDİR ve konum YAKLAŞIKTIR "
                "(gömülü ızgara). 3-4 aylık öngörü gücü sıfıra yakın; "
                "yalnız POZİSYON BOYUTU bağlamıdır.")}


def deger_baglami(carpan_seri, carpan_son, ad="F/K"):
    """Endeks değerlemesinin YÜZDELİK konumu — ZAMANLAMA SİNYALİ DEĞİL."""
    y = _yuzdelik(carpan_seri, carpan_son)
    if y is None:
        return None
    return {
        "olcut": ad, "deger": round(float(carpan_son), 2),
        "yuzdelik": round(y, 3),
        "okuma": (f"{ad} kendi tarihinin %{y*100:.0f}. yüzdeliğinde"),
        "ufuk_uyarisi": (
            "⚠️ BU BİR ZAMANLAMA SİNYALİ DEĞİLDİR. Endeks değerlemesinin "
            "3-4 aylık ufukta öngörü gücü SIFIRA YAKINDIR — CAPE'in 1 "
            "YILLIK ilişkisi bile pratikte sıfırdır (Invesco, 1983-2024). "
            "Anlamlı hale gelmesi ~10 yıl alır (R²~0.43). Goyal-Welch "
            "(RFS 2008): öngörücülerin çoğu örneklem dışında başarısız. "
            "Bu sayı senin 3 aylık kararına neredeyse hiçbir şey "
            "söylemez — POZİSYON BOYUTU için bağlamdır, giriş anı için "
            "değil."),
    }


def yogunlasma_uyarisi(endeks, aday_sayisi=None):
    """Adayların hepsi tek endeksten geliyorsa TEK FAKTÖR bahsi."""
    y = YOGUNLASMA.get(endeks)
    if not y:
        return None
    return {
        "ilk10_agirlik": y["ilk10"], "tarih": y["tarih"],
        "uyari": (
            f"{ENDEKSLER.get(endeks, endeks)} ilk 10 hissesi endeksin "
            f"~%{y['ilk10']*100:.0f}'ini oluşturuyor ({y['tarih']}). "
            f"{y['not']} Adaylarının hepsi bu endeksten geliyorsa "
            f"portföyün büyük ölçüde TEK BİR FAKTÖR bahsidir — "
            f"çeşitlendirme yanılsaması. Kriz anında korelasyon "
            f"yükselir (2008: ortalama 0.39 → 0.62), yani tam da en çok "
            f"ihtiyaç duyduğun anda çeşitlendirme kaybolur."),
    }


def faiz_baglami(endeks, on_yil=None):
    """Faiz bağlamı — SABİT BETA OLARAK KULLANILMAZ."""
    d = FAIZ_DUYARLILIK.get(endeks)
    if d is None:
        return None
    return {
        "duyarlilik": d, "on_yil": on_yil,
        "not": (
            f"{ENDEKSLER.get(endeks, endeks)} uzun vadeli (long-duration) "
            f"bir varlıktır: tarihsel faiz duyarlılığı ~{d:.1f} "
            f"(S&P 500 ~0.72). " + FAIZ_DUYARLILIK["_not"]),
    }


# ── VIX vade yapısı (IVTS) — araştırma A1: en güçlü kısa-ufuk sinyali ─
def vix_vade_yapisi(vix_seri, vix3m_seri):
    """IVTS = VIX / VIX3M. <1.0 contango (olağan, zamanın ~%80-85'i),
    >1.0 backwardation (akut stres). 'Hook' = oran tepe yapıp geri
    dönüyor — tarihsel olarak kapitülasyon dipleri buralarda oluşur.
    Kaynak: VIX-futures basis literatürü (Quantpedia özetleri);
    Yahoo ^VIX3M 2007'den beri. Hook tespiti SEZGİSEL bir kuraldır."""
    try:
        v = pd.Series(vix_seri).dropna()
        u = pd.Series(vix3m_seri).dropna()
        df = pd.concat([v.rename("v"), u.rename("u")], axis=1,
                       join="inner").dropna()
        if len(df) < 120:
            return None
        oran = (df["v"] / df["u"]).dropna()
        son = float(oran.iloc[-1])
        yz = float((oran < son).mean())
        durum = "backwardation" if son > 1.0 else "contango"
        pencere = oran.iloc[-15:]
        tepe = float(pencere.max())
        hook = bool(tepe >= 1.0 and son <= tepe - 0.05 and son < 1.02)
        if hook:
            okuma = ("Oran son 15 günde 1.0 üstüne çıkmış ve belirgin "
                     "GERİ DÖNMÜŞ ('hook'). Tarihsel olarak kapitülasyon "
                     "dipleri bu geri dönüşlerde oluşmuş — kademeli alım "
                     "HIZLANDIRILABİLİR. (Sezgisel kural; kesinlik değil.)")
        elif durum == "backwardation":
            okuma = ("VIX > VIX3M: piyasa YAKIN vadede uzak vadeden çok "
                     "korkuyor — akut stres. Düşen bıçağı tutma; tranşı "
                     "beklet, oranın tepe yapıp dönmesini ('hook') izle.")
        else:
            okuma = ("VIX < VIX3M: olağan (contango) rejim — vade yapısı "
                     "tek başına ayırt edici bilgi taşımıyor.")
        return {"ivts": round(son, 3), "durum": durum,
                "yuzdelik": round(yz, 3), "hook": hook,
                "vix": round(float(df['v'].iloc[-1]), 2),
                "vix3m": round(float(df['u'].iloc[-1]), 2),
                "okuma": okuma,
                "not": ("IVTS = VIX/VIX3M. Zamanın ~%80-85'i contango'dur; "
                        "backwardation nadirdir ve stres işaretidir. Bu bir "
                        "YÖN tahmini değil, rejim/tempo bilgisidir.")}
    except Exception:
        return None


# ── Kredi stresi — araştırma A3: rejim/stres FİLTRESİ ────────────────
def kredi_stresi(oas_seri=None, hyg_seri=None, ief_seri=None, pencere=756):
    """HY kredi spreadi stres filtresi. Tercih: FRED BAMLH0A0HYM2
    (ICE BofA US High Yield OAS, günlük, 1996'dan). Yoksa HYG/IEF
    oranı KABA proxy (faiz süresi etkisi karışır — etiketli).

    DİKKAT: Bu bir HİSSE getiri prediktörü DEĞİL (hisse için belgelenmiş
    OOS R² yok); kredi KOŞULLARININ rejim bilgisidir (Chava-Gallmeyer-
    Park, JME 2015 kredi koşulları-hisse ilişkisini destekler). Skora
    sınırlı katkı yapar, çarpılmaz."""
    try:
        if oas_seri is not None:
            s = pd.Series(oas_seri).dropna()
            if len(s) >= 252:
                son = float(s.iloc[-1])
                ref = s.iloc[-pencere:]
                mu, sd = float(ref.mean()), float(ref.std())
                z = (son - mu) / sd if sd > 0 else 0.0
                d21 = (son - float(s.iloc[-22])) if len(s) >= 22 else None
                durum = ("stres" if z >= 2 else
                         "gergin" if z >= 1 else "sakin")
                return {"kaynak": "FRED HY OAS (BAMLH0A0HYM2)",
                        "deger": round(son, 2), "z": round(z, 2),
                        "d21": (round(d21, 2) if d21 is not None else None),
                        "durum": durum,
                        "okuma": (
                            f"HY spreadi %{son:.2f} — son ~3 yılın "
                            f"{z:+.1f} standart sapması"
                            + (f"; 1 ayda {d21:+.2f} puan" if d21 is not None
                               else "") + ". "
                            + ("Spread hızla açılıyor/yüksek: risk iştahı "
                               "bozuk, tranşları yavaşlat."
                               if z >= 1 else
                               "Kredi piyasası sakin — stres teyidi yok.")),
                        "not": ("Stres FİLTRESİ — hisse yön prediktörü "
                                "değil. Aşırı dar spread (düşük z) 'al' "
                                "sinyali SAYILMAZ (rehavet olabilir).")}
        if hyg_seri is not None and ief_seri is not None:
            h = pd.Series(hyg_seri).dropna()
            e = pd.Series(ief_seri).dropna()
            df = pd.concat([h.rename("h"), e.rename("e")], axis=1,
                           join="inner").dropna()
            if len(df) < 252:
                return None
            oran = (df["h"] / df["e"]).dropna()
            son = float(oran.iloc[-1])
            ref = oran.iloc[-pencere:]
            mu, sd = float(ref.mean()), float(ref.std())
            zo = (son - mu) / sd if sd > 0 else 0.0
            z = -zo  # oran DÜŞERSE kredi stresi → stres-z pozitifleşir
            durum = ("stres" if z >= 2 else
                     "gergin" if z >= 1 else "sakin")
            return {"kaynak": "HYG/IEF oranı (KABA proxy)",
                    "deger": round(son, 4), "z": round(z, 2), "d21": None,
                    "durum": durum,
                    "okuma": (f"HYG/IEF oranı son ~3 yıla göre "
                              f"{zo:+.1f} sd. "
                              + ("Oran belirgin düşük: kredi stresi "
                                 "sinyali — tranşları yavaşlat."
                                 if z >= 1 else
                                 "Kredi proxy'si sakin.")),
                    "not": ("KABA proxy: HYG/IEF faiz süresi etkisini de "
                            "içerir. Asıl kaynak FRED HY OAS'tır; o "
                            "alınamadığı için buna düşüldü.")}
        return None
    except Exception:
        return None


# ── POZİSYON ÇARPANI v2 — KOMPOZİT SKOR (araştırma C) ────────────────
def pozisyon_carpani(rejim, deger=None, vade=None, kredi=None):
    """ASIL ÇIKTI: rejim → pozisyon boyutu çarpanı. v2.

    v1 NEDEN DEĞİŞTİ: eski sürüm faktörleri ÇARPIYORDU ve 0.50-1.50'ye
    kırpıyordu. Stres sinyalleri birbiriyle KORELASYONLUDUR (hepsi aynı
    rejimde tetiklenir); çarpım uçlarda SAHTE kesinlik biriktirir.
    Cederburg-O'Doherty-Wang-Yan (JFE 2020): vol-yönetimli portföyler
    örneklem DIŞINDA sistematik kazandırmıyor — agresif çarpan (>1.30)
    kanıt taşımıyor. Harvey vd. (JPM 2018): vol hedefleme risk
    varlıklarında Sharpe'ı ılımlı iyileştirir, asıl fayda SOL KUYRUK
    kısaltmasıdır.

    v2 YÖNTEM: her alt-sinyal [-1, +1] skora çevrilir, ORTALAMASI
    alınır, çarpan = 1 + 0.30 × kırp(ort, -1, +1) → [0.70, 1.30].
    Sinyaller aynı yöne bakınca etki BÜYÜMEZ, sadece ortalama dolar —
    korelasyonlu sinyalde dürüst davranış budur.
    """
    if not rejim:
        return None
    skorlar, gerekce = {}, []

    # 1) Trend (MA200) — asimetrik: kanıt AŞAĞI koruma yönünde
    #    (Faber 2007: 10ay SMA maks. düşüşü ~%46→<%10; getiri artırmaz)
    if rejim.get("ma200_ustu") is False:
        skorlar["trend"] = -1.0
        gerekce.append("trend: fiyat MA200 ALTINDA (yüksek oynaklık "
                       "rejimi) → −1.0")
    else:
        skorlar["trend"] = 0.25
        gerekce.append("trend: fiyat MA200 üstünde → +0.25")

    # 2) Çekilme kovası — dağılım kayması (T. Rowe Price 1940-2025)
    kv = (rejim.get("cekilme_kova") or {}).get("ad")
    if kv == "ayı bölgesi":
        skorlar["cekilme"] = 1.0
        gerekce.append("çekilme: ayı bölgesi — forward dağılım tarihsel "
                       "olarak LEHTE kaymış → +1.0")
    elif kv == "düzeltme bölgesi":
        skorlar["cekilme"] = 0.5
        gerekce.append("çekilme: düzeltme bölgesi → +0.5")
    else:
        skorlar["cekilme"] = 0.0
        gerekce.append("çekilme: sığ/olağan bölge → 0.0")

    # 3) VIX yüzdelik (Giot 2005: 80.+ yüzdelikte forward getiri
    #    ortalama üstü — risk primi hasadı)
    vy = rejim.get("vix_yuzdelik")
    if vy is not None:
        if vy >= 0.90:
            skorlar["vix"] = 0.5
            gerekce.append("VIX: 90. yüzdelik üstü — risk primi yüksek "
                           "→ +0.5")
        elif vy >= 0.75:
            skorlar["vix"] = -0.5
            gerekce.append("VIX: yüksek ama aşırı değil — belirsizlik "
                           "primi → −0.5")
        else:
            skorlar["vix"] = 0.0
            gerekce.append("VIX: olağan bant → 0.0")

    # 4) VIX vade yapısı (IVTS) — akut stres / dönüş
    if vade:
        if vade.get("hook"):
            skorlar["vade"] = 0.5
            gerekce.append("vade yapısı: backwardation'dan HOOK (geri "
                           "dönüş) → +0.5")
        elif vade.get("durum") == "backwardation":
            skorlar["vade"] = -1.0
            gerekce.append("vade yapısı: BACKWARDATION (akut stres, "
                           "IVTS>1) → −1.0")
        else:
            skorlar["vade"] = 0.0
            gerekce.append("vade yapısı: contango (olağan) → 0.0")

    # 5) Kredi stresi — yalnız CEZALANDIRIR (dar spread 'al' değildir)
    if kredi and kredi.get("z") is not None:
        kz = kredi["z"]
        if kz >= 2:
            skorlar["kredi"] = -1.0
            gerekce.append("kredi: HY spread stres bölgesi (z≥2) → −1.0")
        elif kz >= 1:
            skorlar["kredi"] = -0.5
            gerekce.append("kredi: HY spread gergin (z≥1) → −0.5")
        else:
            skorlar["kredi"] = 0.0
            gerekce.append("kredi: sakin → 0.0")

    # 6) Değerleme (varsa) — yalnız KONUM bilgisi
    if deger and deger.get("yuzdelik") is not None:
        dy = deger["yuzdelik"]
        if dy >= 0.90:
            skorlar["degerleme"] = -0.5
            gerekce.append("değerleme: 90. yüzdelik üstü → −0.5")
        elif dy <= 0.25:
            skorlar["degerleme"] = 0.5
            gerekce.append("değerleme: alt çeyrek → +0.5")
        else:
            skorlar["degerleme"] = 0.0
            gerekce.append("değerleme: orta bant → 0.0")

    ort = float(np.mean(list(skorlar.values())))
    c = 1.0 + 0.30 * max(-1.0, min(1.0, ort))
    c = float(max(0.70, min(1.30, c)))
    return {
        "carpan": round(c, 2), "ort_skor": round(ort, 2),
        "skorlar": {k: round(v, 2) for k, v in skorlar.items()},
        "gerekce": gerekce or ["nötr rejim"],
        "not": (
            f"Normal pozisyonunun ~{c:.2f} katı. Bu bir GİRİŞ SİNYALİ "
            f"DEĞİL, RİSK BÜTÇESİ ayarıdır. v2: alt-sinyaller [-1,+1] "
            f"skorlanıp ORTALANIR (çarpılmaz — korelasyonlu sinyalde "
            f"çarpım sahte kesinlik üretir); aralık [0.70, 1.30] "
            f"(Cederburg vd. 2020: daha agresif çarpanlar örneklem "
            f"dışında kanıt taşımıyor)."),
    }


def piyasa_baglami(endeks, kapanis, vix_seri=None, vix_son=None,
                   carpan_seri=None, carpan_son=None, on_yil=None,
                   vade=None, kredi=None):
    """Tam bağlam paketi."""
    rejim = endeks_rejimi(kapanis, vix_seri, vix_son)
    if not rejim:
        return None
    if carpan_seri is not None and carpan_son is not None:
        deger = deger_baglami(carpan_seri, carpan_son)
    elif carpan_son is not None:
        # O1: elle girilen değer artık ölü değil — gömülü yaklaşık
        # ızgaraya göre konumlanır (pozisyon çarpanı da işler).
        deger = _elle_deger_baglami(carpan_son)
    else:
        deger = None
    return {
        "endeks": endeks, "endeks_adi": ENDEKSLER.get(endeks, endeks),
        "rejim": rejim, "deger": deger,
        "vade": vade, "kredi": kredi,
        "yogunlasma": yogunlasma_uyarisi(endeks),
        "faiz": faiz_baglami(endeks, on_yil),
        "pozisyon": pozisyon_carpani(rejim, deger, vade, kredi),
        "doktrin": (
            "Bu ekran sana NE ZAMAN alacağını SÖYLEMEZ. Endeks "
            "değerlemesinin 3-4 aylık ufukta öngörü gücü sıfıra yakındır. "
            "Söylediği şey: şu an hangi rejimdesin, o rejimde tarihsel "
            "dağılım nasıldı, ve buna göre pozisyonunu ne kadar "
            "büyütüp küçültmelisin."),
        "reddedilenler": [
            "Endekste hedef seviye ('X'e gelince al') — kanıt yok",
            "Genişlik sinyalleri (A-D, %200G üstü) — RED KALDI ama gerekçe güncellendi: nadir sinyaller (örn. Zweig thrust) 6-12 ayda kanıtlı; sorun constituent verisinin ücretsiz olmaması ve çok seyrek tetiklenmesi",
            "CAPE/Buffett zamanlama tetiği — 3 ayda öngörü ~sıfır",
            "Endekse girme/çıkma olayı — etki kayboldu (%7.4 → %0.3)",
            "Sabit faiz betası — ilişki rejime göre işaret değiştiriyor",
            "'Düzeltmeyi bekle' — gelmeyen düzeltmenin fırsat maliyeti",
        ],
    }


# ══════════════════════════════════════════════════════════════════════
# TABAN ORAN MOTORU — "şu andakine benzer yerlerde ne oldu?"
# ══════════════════════════════════════════════════════════════════════
# Kullanıcının ASIL sorusu: "burası almak için güvenli bir yer mi?"
# Dürüst cevap tek bir kelime değil, BİR DAĞILIM: endeks bugünkü
# durumuna benzer yerlerdeyken sonraki 3 ay tarihsel olarak nasıl
# geçmiş — medyanı, çeyreklikleri, ve KÖTÜ senaryosu ne olmuş.
#
# Bu bir tahmin DEĞİL, geçmişin sayımıdır. İki sınırı açıkça yazılır:
#   1) örtüşen pencereler → bağımsız gözlem sayısı göründüğünden AZ
#   2) geçmiş, geleceğe benzemek zorunda değil
# ══════════════════════════════════════════════════════════════════════

UFUK_GUN = 63          # ~3 işlem ayı (kullanıcının ufku)


def _drawdown_serisi(c):
    """Her gün için 52 haftalık zirveden çekilme."""
    zirve = c.rolling(252, min_periods=60).max()
    return 1.0 - c / zirve


def taban_oranlari(kapanis, ufuk=UFUK_GUN, tolerans=0.03, min_n=30):
    """Bugünküne BENZER çekilme seviyelerinde sonraki `ufuk` ne olmuş?

    Yöntem: her tarihsel gün için 52h zirveden çekilme hesaplanır;
    bugünkü çekilmeye ±`tolerans` yakın olan günler seçilir; o günlerden
    sonraki `ufuk` günün getirisi toplanır.

    Dönen dağılım KOŞULLU TABAN ORANIDIR — "şu an güvenli mi" sorusunun
    dürüst karşılığı budur: tek bir evet/hayır değil, bir dağılım.
    """
    try:
        c = pd.Series(kapanis).dropna()
        if len(c) < 252 + ufuk + 60:
            return {"yetersiz": True, "gozlem": len(c),
                    "not": ("Taban oranı için en az ~%d günlük geçmiş "
                            "gerekiyor." % (252 + ufuk + 60))}
        dd = _drawdown_serisi(c)
        bugun_dd = float(dd.iloc[-1])
        ileri = (c.shift(-ufuk) / c - 1.0)

        gecerli = dd.notna() & ileri.notna()
        benzer = gecerli & (dd - bugun_dd).abs().le(tolerans)
        # son `ufuk` gün ileri getiri taşımaz
        benzer.iloc[-ufuk:] = False

        ornek = ileri[benzer].dropna()
        tum = ileri[gecerli].dropna()
        if len(ornek) < min_n:
            return {"yetersiz": True, "gozlem": int(len(ornek)),
                    "bugun_cekilme": round(bugun_dd, 4),
                    "not": (f"Bugünkü çekilme seviyesine (%{bugun_dd*100:.1f}) "
                            f"benzer yalnız {len(ornek)} tarihsel gün var — "
                            f"taban oranı çıkarmak için yetersiz. Bu, "
                            f"endeksin şu an ALIŞILMADIK bir yerde olduğu "
                            f"anlamına da gelir.")}

        q = lambda s, p: float(np.percentile(s, p))
        # bağımsız gözlem ~ örneklem / ufuk (örtüşme düzeltmesi)
        etkin_n = max(1, int(len(ornek) / ufuk))
        return {
            "yetersiz": False,
            "bugun_cekilme": round(bugun_dd, 4),
            "tolerans": tolerans, "ufuk_gun": ufuk,
            "gozlem": int(len(ornek)), "etkin_gozlem": etkin_n,
            "medyan": round(float(np.median(ornek)), 4),
            "ortalama": round(float(np.mean(ornek)), 4),
            "p10": round(q(ornek, 10), 4), "p25": round(q(ornek, 25), 4),
            "p75": round(q(ornek, 75), 4), "p90": round(q(ornek, 90), 4),
            "pozitif_oran": round(float((ornek > 0).mean()), 3),
            "kotu_oran": round(float((ornek < -0.10).mean()), 3),
            "kosulsuz_medyan": round(float(np.median(tum)), 4),
            "kosulsuz_pozitif": round(float((tum > 0).mean()), 3),
            "not": (
                f"Endeks bugünküne benzer bir çekilmedeyken "
                f"(%{bugun_dd*100:.1f} ±%{tolerans*100:.0f}) sonraki "
                f"{ufuk} işlem gününde tarihsel olarak ne olduğunun "
                f"SAYIMIDIR — tahmin değildir."),
            "sinir": (
                f"⚠️ İKİ SINIR: (1) {len(ornek)} gözlem ÖRTÜŞEN "
                f"pencerelerden geliyor; bağımsız gözlem sayısı ~"
                f"{etkin_n} kadardır, yani güven aralıkları göründüğünden "
                f"GENİŞTİR. (2) Bu dağılım birkaç tarihsel episoda hâkim "
                f"olabilir ve gelecek geçmişe benzemek zorunda değildir."),
        }
    except Exception as e:
        return {"yetersiz": True, "not": f"Hesaplanamadı: {type(e).__name__}"}


def guvenli_mi(taban):
    """'Burası güvenli mi?' sorusuna DÜRÜST cevap — evet/hayır DEĞİL."""
    if not taban or taban.get("yetersiz"):
        return None
    m, poz = taban["medyan"], taban["pozitif_oran"]
    p10, kotu = taban["p10"], taban["kotu_oran"]
    km, kp = taban["kosulsuz_medyan"], taban["kosulsuz_pozitif"]
    fark = m - km

    if fark > 0.02:
        yon = "koşulsuz ortalamadan DAHA İYİ"
    elif fark < -0.02:
        yon = "koşulsuz ortalamadan DAHA KÖTÜ"
    else:
        yon = "koşulsuz ortalamayla neredeyse AYNI"

    return {
        "ozet": (
            f"Endeks bugünküne benzer yerlerdeyken sonraki 3 ay: medyan "
            f"%{m*100:+.1f}, vakaların %{poz*100:.0f}'inde pozitif. Bu, "
            f"{yon} (koşulsuz medyan %{km*100:+.1f}, pozitif oran "
            f"%{kp*100:.0f})."),
        "kotu_senaryo": (
            f"AMA aynı durumlarda en kötü %10'da getiri %{p10*100:.1f} "
            f"olmuş ve vakaların %{kotu*100:.0f}'inde 3 ayda %10'dan "
            f"fazla kayıp görülmüş. 'Güvenli' kelimesi burada "
            f"kullanılamaz — kullanılabilecek kelime OLASILIK DAĞILIMI."),
        "hukum": (
            "Bu seviye, tarihsel dağılımı BELİRGİN biçimde lehe kaydırmış."
            if fark > 0.02 else
            "Bu seviye, tarihsel dağılımı belirgin biçimde DEĞİŞTİRMEMİŞ — "
            "yani 'iyi bir yer' demek için elde kanıt yok."
            if abs(fark) <= 0.02 else
            "Bu seviyede tarihsel dağılım ALEYHE kaymış."),
    }

# ── Hisse–endeks ilişkisi — araştırma E: 'ek güven' bu hisse için ne kadar?
def hisse_endeks_iliskisi(hisse_kapanis, endeks_kapanis, min_gun=120):
    """Beta, korelasyon, R² — endeks ekranının BU hisse için gerçek
    değeri. Tipik hissede piyasa faktörü getiri varyansının kabaca
    üçte biri-yarısını açıklar; yeni IPO'larda çok daha azını.
    R² düşükse endeks bağlamı yalnız genel risk-iştahı filtresidir."""
    try:
        h = pd.Series(hisse_kapanis).dropna()
        e = pd.Series(endeks_kapanis).dropna()
        df = pd.concat([h.rename("h"), e.rename("e")], axis=1,
                       join="inner").dropna()
        rh = df["h"].pct_change().dropna()
        re = df["e"].pct_change().dropna()
        n = int(min(len(rh), len(re)))
        if n < min_gun:
            return {"yetersiz": True, "gun": n,
                    "not": (f"Ortak yalnız {n} işlem günü var "
                            f"(en az {min_gun} gerekir) — beta/R² "
                            f"güvenilir hesaplanamaz. Yeni halka arzlarda "
                            f"bu OLAĞANDIR; endeks bağlamına değil, "
                            f"şirkete özgü habere bak.")}
        rh, re = rh.iloc[-n:], re.iloc[-n:]
        var_e = float(re.var())
        beta = float(rh.cov(re) / var_e) if var_e > 0 else None
        r = float(rh.corr(re))
        r2 = r * r if r == r else None  # NaN koruması
        if r2 is None or beta is None:
            return {"yetersiz": True, "gun": n,
                    "not": "Beta/R² hesaplanamadı (varyans/korelasyon sorunu)."}
        if r2 >= 0.50:
            kova, yorum = "güçlü", (
                "Endeks bağlamı bu hisse için ANLAMLI: getirisinin "
                "yarıdan fazlası piyasa faktörüyle hareket ediyor. "
                "Bu ekranın rejim/tempo çıktısı burada gerçek 'ek "
                "güven' sağlar.")
        elif r2 >= 0.30:
            kova, yorum = "kısmi", (
                "Endeks bağlamı KISMEN geçerli: getirinin kabaca "
                "üçte biri-yarısı piyasadan geliyor. Rejim ekranını "
                "boyut/tempo için kullan; yönü yine şirket belirler.")
        else:
            kova, yorum = "zayıf / idiosinkratik", (
                "Bu hisse büyük ölçüde KENDİ haberiyle oynuyor — "
                "endeks yukarı gitse bile hisse düşebilir. Bu ekran "
                "burada yalnız GENEL risk-iştahı filtresidir; hisseye "
                "özgü sinyal DEĞİLDİR. 'Ek güven' payını küçük tut.")
        ipo = n < 250
        return {"yetersiz": False, "gun": n, "beta": round(beta, 2),
                "korelasyon": round(r, 2), "r2": round(r2, 3),
                "kova": kova, "yorum": yorum, "ipo_uyari": ipo,
                "ipo_not": (
                    "⚠️ 250 günden az geçmiş (yeni halka arz olabilir): "
                    "beta/R² henüz OTURMAMIŞTIR ve tarihsel taban "
                    "oranları bu hisseye UYGULANAMAZ." if ipo else None)}
    except Exception:
        return {"yetersiz": True, "gun": 0, "not": "Hesaplanamadı."}


# ── Tranş planı — araştırma B: kanıt LSI lehine; alt tranşlar risk kontrolü
def trans_plani(kapanis, carpan=1.0, esikler=(0.05, 0.10),
                ufuk=UFUK_GUN, tolerans=0.03, min_n=30):
    """Bugünkü rejime göre kademeli alım planı + her alt tranşın
    TARİHSEL dolum olasılığı ve dolmama fırsat maliyeti.

    Dolum olasılığı: bugünkü çekilmeye benzer (±tolerans) tarihsel
    günlerden sonraki `ufuk` işlem günü içinde fiyatın o günkü
    kapanışın en az −e% altına inme oranı. Fırsat maliyeti: tranşın
    DOLMADIĞI benzer günlerde `ufuk` günlük medyan getiri — o parayı
    bekleterek tarihsel olarak neyi kaçırırdın.

    ÇERÇEVE (Vanguard 2023): peşin alım ~2/3 oranda kademeliden iyi;
    ilk tranş bu yüzden ŞİMDİ. Alt tranşlar getiri optimizasyonu
    değil, DAVRANIŞSAL/risk kontrolüdür — yıllarca dolmayabilirler
    (tarihte 7 yıl düzeltmesiz dönem var)."""
    try:
        c = pd.Series(kapanis).dropna()
        if len(c) < 252 + ufuk + 60:
            return {"yetersiz": True,
                    "not": "Tranş planı için yeterli tarih yok."}
        dd = _drawdown_serisi(c)
        bugun_dd = float(dd.iloc[-1])
        gecerli = dd.notna()
        benzer = gecerli & (dd - bugun_dd).abs().le(tolerans)
        benzer.iloc[-ufuk:] = False
        idx = np.where(benzer.values)[0]
        arr = c.values.astype(float)
        satirlar, n_kul = [], 0
        for e in esikler:
            dolan, firsat, toplam = 0, [], 0
            for i in idx:
                fut = arr[i + 1:i + 1 + ufuk]
                if len(fut) < ufuk:
                    continue
                toplam += 1
                if fut.min() / arr[i] - 1.0 <= -e:
                    dolan += 1
                else:
                    firsat.append(arr[i + ufuk] / arr[i] - 1.0)
            if toplam < min_n:
                return {"yetersiz": True,
                        "not": (f"Bugünkü çekilmeye benzer yeterli "
                                f"tarihsel gün yok ({toplam} < {min_n}) — "
                                f"dolum olasılığı çıkarılamaz.")}
            n_kul = toplam
            satirlar.append({
                "esik": e,
                "seviye": round(float(arr[-1]) * (1 - e), 2),
                "dolum_orani": round(dolan / toplam, 3),
                "firsat_medyan": (round(float(np.median(firsat)), 4)
                                  if firsat else None)})
        if carpan >= 1.10:
            pay = (60, 25, 15)
        elif carpan <= 0.90:
            pay = (40, 30, 30)
        else:
            pay = (50, 30, 20)
        etkin = max(1, int(n_kul / ufuk))
        return {"yetersiz": False, "pay": pay, "satirlar": satirlar,
                "gozlem": n_kul, "etkin_gozlem": etkin,
                "ufuk_gun": ufuk, "bugun_cekilme": round(bugun_dd, 4),
                "not": (
                    f"İlk tranş (%{pay[0]}) ŞİMDİ — kanıt peşin alım "
                    f"lehine (Vanguard 2023: LSI ~2/3 oranda kazanır). "
                    f"Alt tranşlar risk/davranış kontrolü içindir. "
                    f"Eşikler ENDEKS üzerinden ölçülür; düşük R²'li "
                    f"hissede hisse, endeks düşmeden de düşebilir."),
                "sinir": (
                    f"⚠️ Dolum oranları örtüşen pencerelerden ({n_kul} "
                    f"gözlem ≈ {etkin} bağımsız) — göründüğünden az "
                    f"bilgi taşır. Alt tranş dolmayabilir; benzer "
                    f"günlerde dolmadığında {ufuk} günlük medyan "
                    f"getiri tabloda ('bekleme maliyeti').")}
    except Exception as ex:
        return {"yetersiz": True, "not": f"Hesaplanamadı: {type(ex).__name__}"}

# ═══════════════════════════════════════════════════════════════════════
# KIYAS MOTORU v2 — 3-4 AYLIK UFUKTA BAŞA BAŞ KARŞILAŞTIRMA
# ═══════════════════════════════════════════════════════════════════════
# Eski `rakip_ozet` 27 metriği 4 eksende yüzdeliğe çevirip tek sayı
# üretiyordu. İki sorunu vardı: (1) N=2-8'de yüzdelik SAHTE KESİNLİKTİR,
# (2) metriklerin çoğu (değer, kalite, ROIC) 3-4 AYLIK ufukta getiri
# ayrıştırmaz — onlar 1-3 yıllık fenomenlerdir.
#
# Bu motor yalnız UFKA UYAN sinyallerle sıralar, sıralamayı KADEME olarak
# sunar, ayırt edilemeyeni açıkça öyle der ve vetoyu sıralamadan AYRI
# tutar. Eski `rakip_ozet` KALDIRILMADI — metrik tablosu hâlâ değerli;
# yanına bu motor kondu.
# ═══════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────
# 0) SABİTLER
# ──────────────────────────────────────────────────────────────────────
IC_VARSAYIM = 0.06          # harmanlanmış sinyalin gerçekçi IC'si
BERABERLIK_SD = 0.35        # bu farkın altı "ayırt edilemez" (araştırma: 0.3-0.5)
KADEME_ORTUSME = 0.25       # bootstrap sıra dağılımı bu kadar örtüşürse aynı kademe
BOOTSTRAP_N = 1000

# Kesitsel referans dağılımlar — z-skor ADAY KÜMESİNE göre değil GENİŞ
# EVRENE göre alınmalı (N=3'te aday-içi z-skor sahte ayrışma üretir).
# Canlı örneklem yoksa bu yaklaşık parametreler kullanılır ve çıktıda
# AÇIKÇA etiketlenir.
REFERANS = {
    "momentum_12_1": {"ort": 0.08, "sd": 0.35,
                      "kaynak": "ABD kesitsel 12-1 getiri, yaklaşık"},
    "zirve_yakinlik": {"ort": 0.82, "sd": 0.15,
                       "kaynak": "fiyat/52h-zirve oranı, yaklaşık"},
    "revizyon_orani": {"ort": 0.00, "sd": 0.45,
                       "kaynak": "(yukarı−aşağı)/(yukarı+aşağı), yaklaşık"},
    "acik_satis": {"ort": 0.045, "sd": 0.050,
                   "kaynak": "float'ın %'si, sağa çarpık — yaklaşık"},
}

# Sinyal ağırlıkları. Kanıt gücüne göre YARGISAL; eşit ağırlık da
# savunulabilir (DeMiguel-Garlappi-Uppal mantığı) — duyarlılık gösterilir.
SINYAL_AGIRLIK = {
    "momentum_12_1": 1.0,
    "revizyon_orani": 1.0,
    "zirve_yakinlik": 0.7,
    "acik_satis": 0.6,
}
SINYAL_YON = {                # +1: yüksek iyi · −1: yüksek kötü
    "momentum_12_1": +1, "revizyon_orani": +1,
    "zirve_yakinlik": +1, "acik_satis": -1,
}
SINYAL_ADI = {
    "momentum_12_1": "12-1 momentum",
    "revizyon_orani": "analist revizyon genişliği",
    "zirve_yakinlik": "52h zirveye yakınlık",
    "acik_satis": "açığa satış (ters)",
}


def ikili_isabet(ic=IC_VARSAYIM):
    """IC'den ikili sıralama isabetine — tasarımın dürüstlük çıpası."""
    return 0.5 + math.asin(max(-1.0, min(1.0, ic))) / math.pi


def getiri_ustunluk_olasiligi(dz, ic=IC_VARSAYIM):
    """P(A gerçekten B'den fazla kazanır) — sinyal farkına KALİBRE.

    ⚠️ BU, BOOTSTRAP P(#1) İLE AYNI ŞEY DEĞİLDİR ve karıştırılırsa
    araç yalan söyler:
      · bootstrap P(#1) → "sinyalleri doğru ölçtüysem sıralama bu"
      · bu fonksiyon    → "sinyal ile GELECEK GETİRİ arasındaki zayıf
                           bağ göz önüne alınınca gerçekten kazanır mı"

    Türetme (iki değişkenli normal, standartlaştırılmış z ve r):
        r | z ~ N(ρz, 1−ρ²)
        P(r_A > r_B) = Φ( ρ·Δz / √(2(1−ρ²)) )
    ÖLÇÜM (ρ=0.06): Δz=1.0 → %51.7 · Δz=2.0 → %53.4 · Δz=3.2 → %55.4
    Yani DEV bir sinyal farkı bile getiri üstünlüğünü ancak %55'e taşır.
    Bootstrap aynı durumda %100 der — o yüzden ikisi AYRI gösterilir.
    """
    if dz is None:
        return None
    ic = max(1e-6, min(0.99, float(ic)))
    z = ic * float(dz) / math.sqrt(2.0 * (1.0 - ic ** 2))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ──────────────────────────────────────────────────────────────────────
# 1) VETO KATMANI — sıralamadan ÖNCE, telafi EDİLEMEZ
# ──────────────────────────────────────────────────────────────────────
def veto_denetimi(h):
    """Adayı diskalifiye eden / uyaran bulgular.

    Doktrin: veto TELAFİ EDİLEMEZ. Güçlü momentum, iflas riskini ya da
    yeniden düzenleme bildirimini SATIN ALAMAZ. Bu, derin motorun
    'kalite pahalılığı mazur göstermez' kuralının kardeşidir.

    Kanıt notu: Beneish M-skorunun holdout yanlış-alarm oranı ~%17.5'tir
    (Beneish 1999) — bu yüzden SERT veto değil İNCELEME bayrağıdır.
    """
    sert, yumusak, takvim = [], [], []

    if h.get("restatement_8k"):
        sert.append("SEC 8-K Item 4.02 — önceki finansallar GÜVENİLMEZ "
                    "ilan edilmiş. Bu ufukta tartışılacak bir tez yok.")
    if h.get("going_concern"):
        sert.append("Denetçi 'işletmenin sürekliliği' şüphesi bildirmiş.")
    if h.get("denetci_degisimi"):
        yumusak.append("Denetçi değişimi (8-K 4.01) — sebebini doğrula.")

    z = h.get("altman")
    if z is not None and not h.get("is_financial") and z < 1.81:
        sert.append(f"Altman Z {z:.2f} — iflas riski bölgesi. "
                    f"Campbell-Hilscher-Szilagyi (2008): sıkıntılı hisseler "
                    f"anormal DÜŞÜK getiri veriyor ve oynaklık yükseldiğinde "
                    f"farkı AÇILIYOR.")

    bm = h.get("beneish")
    if bm is not None and bm > -1.78:
        yumusak.append(f"Beneish M {bm:.2f} > −1.78 — kâr manipülasyonu "
                       f"OLASILIĞI eşiğin üstünde. DİKKAT: modelin yanlış "
                       f"alarm oranı ~%17.5, yani bu bir kanıt değil "
                       f"İNCELEME çağrısıdır.")

    fcf = h.get("fcf")
    if fcf is not None and fcf <= 0:
        yumusak.append("Serbest nakit akışı negatif.")

    sh = h.get("acik_satis")
    if sh is not None and sh > 0.20:
        yumusak.append(f"Float'ın %{sh*100:.0f}'i açığa satılmış. Kanıt bu "
                       f"ufukta ALEYHTE (Rapach vd. 2016) ama aşırı "
                       f"kalabalıkta sıkışma riski iki yönlüdür.")

    lik = h.get("gunluk_hacim_usd")
    if lik is not None:
        # ARAŞTIRMA TURU A3: mutlak $5M tabanı perakende ölçek için fazla
        # katıydı — FIM (2018, $1.7T canlı işlem verisi): %1 ADV'lik
        # işlemin fiyat etkisi ~9bp; $5-50K'lık perakende pozisyon $5M
        # hacimde zaten %0.1-1 katılımdır. Asıl maliyet SPREAD ve o,
        # tipik olarak <$1M dolar-hacimde ısırır. Kademeli eşik:
        if lik < 1e5:
            sert.append(f"Günlük işlem hacmi ~${lik/1e3:.0f}K — MİKRO "
                        f"likidite; spread tek başına 3 aylık kenarı "
                        f"yiyebilir ve çıkışta fiyat garantisi yoktur.")
        elif lik < 5e5:
            yumusak.append(f"Günlük hacim ~${lik/1e6:.2f}M — DAR; gidiş-"
                           f"dönüş maliyeti (spread ağırlıklı) kenarı "
                           f"anlamlı yiyebilir. Limit emir + bölerek gir.")
        elif lik < 2e6:
            yumusak.append(f"Günlük hacim ~${lik/1e6:.1f}M — orta; piyasa "
                           f"emri yerine limit emir kullan.")

    kg = h.get("kazanc_gun_kaldi")
    if kg is not None and 0 <= kg <= 100:
        takvim.append(f"{kg} gün sonra KAZANÇ açıklaması — tutma "
                      f"penceresinin İÇİNDE. Bu bir yön sinyali değil, "
                      f"ikili olay riskidir; pozisyonu ona göre boyutlandır.")

    # ── Denetim Bulgu 2 (görünürlük): girdisi gelmeyen veto SESSİZCE
    # "temiz" görünüyordu — boş liste ile "denetlenemedi" aynı şey değil.
    # Hangi kontrollerin bu veri kümesinde ÇALIŞAMADIĞI açıkça raporlanır.
    _girdi_haritasi = {
        "restatement_8k": "8-K 4.02 restatement (SERT veto)",
        "going_concern": "going-concern (SERT veto)",
        "denetci_degisimi": "denetçi değişimi uyarısı",
        "altman": "Altman iflas-bölgesi (SERT veto)",
        "beneish": "Beneish manipülasyon uyarısı",
        "gunluk_hacim_usd": "likidite uyarısı",
        "kazanc_gun_kaldi": "kazanç-takvimi uyarısı",
    }
    kapsanmayan = [ad for k, ad in _girdi_haritasi.items()
                   if h.get(k) is None]
    return {"sert": sert, "yumusak": yumusak, "takvim": takvim,
            "diskalifiye": bool(sert), "kapsanmayan": kapsanmayan}


# ──────────────────────────────────────────────────────────────────────
# 2) SEKTÖR-NÖTR Z-SKOR
# ──────────────────────────────────────────────────────────────────────
def z_skor(deger, sinyal, referans=None):
    """Sinyali GENİŞ EVRENE göre standartlaştır.

    KRİTİK: z-skor aday kümesine göre alınırsa (N=2-8) sahte ayrışma
    üretir — 3 hissede yüzdelikler zorunlu olarak 0/50/100 çıkar ve
    aralarındaki fark ne olursa olsun aynı görünür.
    """
    if deger is None:
        return None
    r = (referans or REFERANS).get(sinyal)
    if not r or not r.get("sd"):
        return None
    z = (float(deger) - r["ort"]) / r["sd"]
    return float(max(-3.0, min(3.0, z))) * SINYAL_YON.get(sinyal, 1)


def aday_z_skorlari(h, referans=None):
    """Bir adayın tüm ufka-uyan sinyallerinin z-skorları."""
    out, kapsam = {}, 0
    for s in SINYAL_AGIRLIK:
        z = z_skor(h.get(s), s, referans)
        out[s] = z
        kapsam += int(z is not None)
    return out, kapsam


def bilesik_skor(zler):
    """Ağırlıklı bileşik — YALNIZ mevcut sinyaller üzerinden."""
    p = w = 0.0
    for s, z in zler.items():
        if z is None:
            continue
        a = SINYAL_AGIRLIK.get(s, 1.0)
        p += z * a
        w += a
    return (p / w) if w else None


# ──────────────────────────────────────────────────────────────────────
# 3) İKİLİ KARŞILAŞTIRMA
# ──────────────────────────────────────────────────────────────────────
def ikili_karsilastir(a_z, b_z, esik=BERABERLIK_SD):
    """Sinyal sinyal A mı B mi — farkı eşiğin altındaysa BERABERE.

    Beraberlik eşiği kasıtlıdır: 0.35 SD'lik bir fark, bu ufukta
    gürültüden ayırt edilemez. 'A 0.02 önde' demek sahte kesinliktir.
    """
    kazanan = {"a": 0, "b": 0, "berabere": 0}
    detay = []
    for s in SINYAL_AGIRLIK:
        za, zb = a_z.get(s), b_z.get(s)
        if za is None or zb is None:
            detay.append({"sinyal": s, "ad": SINYAL_ADI[s],
                          "sonuc": "veri yok", "fark": None})
            continue
        d = za - zb
        if abs(d) < esik:
            kazanan["berabere"] += 1
            son = "berabere"
        elif d > 0:
            kazanan["a"] += 1
            son = "A"
        else:
            kazanan["b"] += 1
            son = "B"
        detay.append({"sinyal": s, "ad": SINYAL_ADI[s], "sonuc": son,
                      "fark": round(d, 2)})
    return kazanan, detay


def copeland_borda(adaylar, zler, esik=BERABERLIK_SD):
    """İki bağımsız sıralama kuralı — ANLAŞMAZLIK kırılganlık sinyalidir.

    Borda: konumsal toplam (Heilman: gürültüye en dayanıklı konumsal kural)
    Copeland: ikili galibiyet − mağlubiyet
    İkisi farklı sıra veriyorsa sıralama SAĞLAM DEĞİLDİR ve kademe
    birleştirilir.
    """
    n = len(adaylar)
    cope = {t: 0.0 for t in adaylar}
    for i in range(n):
        for j in range(i + 1, n):
            ta, tb = adaylar[i], adaylar[j]
            k, _ = ikili_karsilastir(zler[ta], zler[tb], esik)
            if k["a"] > k["b"]:
                cope[ta] += 1; cope[tb] -= 1
            elif k["b"] > k["a"]:
                cope[tb] += 1; cope[ta] -= 1
    bil = {t: bilesik_skor(zler[t]) for t in adaylar}
    gecerli = [t for t in adaylar if bil[t] is not None]
    sirali = sorted(gecerli, key=lambda t: bil[t], reverse=True)
    borda = {t: (len(sirali) - i) for i, t in enumerate(sirali)}
    for t in adaylar:
        borda.setdefault(t, 0)
    b_sira = sorted(adaylar, key=lambda t: -borda[t])
    c_sira = sorted(adaylar, key=lambda t: -cope[t])
    return {"borda": borda, "copeland": cope,
            "borda_sira": b_sira, "copeland_sira": c_sira,
            "anlasiyor": b_sira == c_sira,
            "bilesik": bil}


# ──────────────────────────────────────────────────────────────────────
# 4) BOOTSTRAP SIRA DAĞILIMI
# ──────────────────────────────────────────────────────────────────────
def bootstrap_sira(adaylar, zler, n=BOOTSTRAP_N, seed=42,
                   sinyal_gurultu=0.5):
    """Sinyal gürültüsü altında her adayın SIRA DAĞILIMI.

    Neden: kurumsal sıralama literatürü (Goldstein-Spiegelhalter 1996)
    sıra belirsizliği gösterilmeden lig tablosu yayımlanmamasını söyler;
    pratikte yalnız uçlar güvenilir sıralanır. Burada da öyle: P(#1)
    çubukları DÜZ çıkarsa, bu bir kusur değil BULGUDUR.
    """
    rng = np.random.default_rng(seed)
    sayac = {t: np.zeros(len(adaylar), dtype=int) for t in adaylar}
    for _ in range(n):
        puan = {}
        for t in adaylar:
            z = {s: (None if v is None else v + rng.normal(0, sinyal_gurultu))
                 for s, v in zler[t].items()}
            puan[t] = bilesik_skor(z)
        gec = [t for t in adaylar if puan[t] is not None]
        if not gec:
            continue
        sirali = sorted(gec, key=lambda t: puan[t], reverse=True)
        for i, t in enumerate(sirali):
            sayac[t][i] += 1
    out = {}
    for t in adaylar:
        top = sayac[t].sum()
        if not top:
            out[t] = {"p_birinci": None, "sira_dagilim": None,
                      "sira_ga95": None, "ortalama_sira": None}
            continue
        d = sayac[t] / top
        kum = np.cumsum(d)
        alt = int(np.searchsorted(kum, 0.025) + 1)
        ust = int(np.searchsorted(kum, 0.975) + 1)
        out[t] = {"p_birinci": round(float(d[0]), 3),
                  "sira_dagilim": [round(float(x), 3) for x in d],
                  "sira_ga95": (alt, min(ust, len(adaylar))),
                  "ortalama_sira": round(
                      float(sum((i + 1) * d[i] for i in range(len(d)))), 2)}
    return out


# ──────────────────────────────────────────────────────────────────────
# 5) KADEMELEME — kesin sıra yerine
# ──────────────────────────────────────────────────────────────────────
def kademele(adaylar, bil, boot, ortusme=KADEME_ORTUSME):
    """Ayırt edilemeyen adayları AYNI kademeye topla.

    Kural: iki adayın bootstrap sıra güven aralıkları belirgin
    örtüşüyorsa ya da bileşik skor farkı beraberlik eşiğinin altındaysa
    aynı kademedeler. Kesin 1-2-3, veri bunu taşımıyorken YALANDIR.
    """
    gec = [t for t in adaylar if bil.get(t) is not None]
    if not gec:
        return []
    sirali = sorted(gec, key=lambda t: bil[t], reverse=True)
    kademeler, akt = [], [sirali[0]]
    for onc, t in zip(sirali, sirali[1:]):
        fark = abs(bil[onc] - bil[t])
        ga1 = (boot.get(onc) or {}).get("sira_ga95")
        ga2 = (boot.get(t) or {}).get("sira_ga95")
        ort = False
        if ga1 and ga2:
            kesisim = min(ga1[1], ga2[1]) - max(ga1[0], ga2[0]) + 1
            birlesim = max(ga1[1], ga2[1]) - min(ga1[0], ga2[0]) + 1
            ort = (kesisim / birlesim) > ortusme if birlesim > 0 else False
        if fark < BERABERLIK_SD or ort:
            akt.append(t)
        else:
            kademeler.append(akt)
            akt = [t]
    kademeler.append(akt)
    return kademeler


# ──────────────────────────────────────────────────────────────────────
# 6) SENARYO — hangi koşulda hangisi
# ──────────────────────────────────────────────────────────────────────
def senaryo_farki(getiriler, faktorler, min_gozlem=100):
    """Adayların faktör duyarlılık FARKLARI.

    Mutlak beta yerine FARK raporlanır: farklar daha kararlıdır ve
    doğrudan "petrol yükselirse hangisi" sorusuna cevap verir.
    Uyarılar: betalar zamanla oynar, kısa örneklemde standart hatalar
    büyüktür, faktörler arası çoklu doğrusal bağlantı katsayıları
    güvenilmez kılar.
    """
    if not getiriler or not faktorler:
        return None
    adlar = list(getiriler)
    n = min(len(v) for v in list(getiriler.values()) + list(faktorler.values()))
    if n < min_gozlem:
        return {"yetersiz": True, "gozlem": n, "gereken": min_gozlem,
                "not": (f"Yalnız {n} gözlem var; faktör duyarlılığı için "
                        f"en az {min_gozlem} (≈2 yıl haftalık) gerekir. "
                        f"Daha azıyla hesaplanan beta gürültüdür.")}
    F = np.column_stack([np.asarray(faktorler[f][-n:], float)
                         for f in faktorler])
    F = np.column_stack([np.ones(n), F])
    beta = {}
    for t in adlar:
        y = np.asarray(getiriler[t][-n:], float)
        try:
            b, *_ = np.linalg.lstsq(F, y, rcond=None)
            beta[t] = {f: round(float(b[i + 1]), 3)
                       for i, f in enumerate(faktorler)}
        except Exception:
            beta[t] = None
    farklar = {}
    for i in range(len(adlar)):
        for j in range(i + 1, len(adlar)):
            a, b_ = adlar[i], adlar[j]
            if not beta.get(a) or not beta.get(b_):
                continue
            farklar[f"{a} − {b_}"] = {
                f: round(beta[a][f] - beta[b_][f], 3) for f in faktorler}
    return {"beta": beta, "farklar": farklar, "gozlem": n,
            "not": ("Betalar ZAMANLA OYNAR ve bu örneklemde standart "
                    "hataları büyüktür; faktörler birbirine bağımlıysa "
                    "tek tek katsayılar güvenilmezdir. Bunları NOKTA "
                    "TAHMİN değil, YÖN göstergesi olarak oku. Senaryoyu "
                    "kendi lehine ayarlayıp ona göre işlem yapma — "
                    "kontrol yanılsaması üretir.")}


# ──────────────────────────────────────────────────────────────────────
# 7) KANAAT BAZLI BOYUTLANDIRMA
# ──────────────────────────────────────────────────────────────────────
def boyutlandirma(adaylar, boot, vetolar, tavan=0.40):
    """Tek kazanan seçmek yerine KANAATE göre dağıt.

    Gerekçe: ikili isabet ~%52 iken tüm parayı 'birinci'ye koymak,
    kenarın taşıyabileceğinden fazla iddiadır. Küçük kenarı bağımsız
    bahislere yaymak (Grinold'un temel yasası) daha savunulabilir.
    """
    pay = {}
    for t in adaylar:
        if (vetolar.get(t) or {}).get("diskalifiye"):
            pay[t] = 0.0
            continue
        p = (boot.get(t) or {}).get("p_birinci")
        pay[t] = 0.0 if p is None else float(p)
    top = sum(pay.values())
    if top <= 0:
        # hiçbir sinyal yok ama vetosuz aday var → EŞİT dağıt
        temiz = [t for t in adaylar
                 if not (vetolar.get(t) or {}).get("diskalifiye")]
        return ({t: (round(1.0 / len(temiz), 3) if t in temiz else 0.0)
                 for t in adaylar} if temiz else
                {t: 0.0 for t in adaylar})
    pay = {t: v / top for t, v in pay.items()}

    # ── TAVAN, ADAY SAYISINA DUYARLI OLMALI ──────────────────────────
    # Sabit %40 tavanı 2 vetosuz adayda MATEMATİKSEL OLARAK İMKÂNSIZDIR
    # (2×0.40 = 0.80) ve kalan %20 sessizce buharlaşırdı — testte
    # yakalandı. Tavan asla 1/N'in altına inemez: 2 adayda doğal taban
    # eşit ağırlıktır (%50), 3'te %33.3.
    n_temiz = sum(1 for t in adaylar
                  if not (vetolar.get(t) or {}).get("diskalifiye"))
    tavan_etkin = max(float(tavan), 1.0 / max(n_temiz, 1))

    for _ in range(8):
        asan = {t: v for t, v in pay.items() if v > tavan_etkin + 1e-9}
        if not asan:
            break
        artik = sum(v - tavan_etkin for v in asan.values())
        for t in asan:
            pay[t] = tavan_etkin
        alt = [t for t, v in pay.items()
               if v < tavan_etkin - 1e-9
               and not (vetolar.get(t) or {}).get("diskalifiye")]
        if not alt:
            break
        for t in alt:
            pay[t] += artik / len(alt)
    # yuvarlama artığını en büyük paya ver → toplam tam 1.000
    yuv = {t: round(v, 3) for t, v in pay.items()}
    fark = round(1.0 - sum(yuv.values()), 3)
    if abs(fark) >= 0.001:
        en_buyuk = max(yuv, key=lambda t: yuv[t])
        if yuv[en_buyuk] > 0:
            yuv[en_buyuk] = round(yuv[en_buyuk] + fark, 3)
    return yuv


# ──────────────────────────────────────────────────────────────────────
# 8) ORKESTRA
# ──────────────────────────────────────────────────────────────────────
def kiyas_motoru(hisseler, referans=None, seed=42):
    """hisseler: {ticker: {sinyaller + veto girdileri}} → tam kıyas."""
    adaylar = list(hisseler)
    if len(adaylar) < 2:
        return None

    vetolar = {t: veto_denetimi(hisseler[t]) for t in adaylar}
    zler, kapsam = {}, {}
    for t in adaylar:
        zler[t], kapsam[t] = aday_z_skorlari(hisseler[t], referans)

    yarisan = [t for t in adaylar if not vetolar[t]["diskalifiye"]]
    if len(yarisan) < 2:
        return {"adaylar": adaylar, "vetolar": vetolar, "z": zler,
                "kapsam": kapsam, "yarisan": yarisan,
                "hukum": ("Veto sonrası karşılaştırılacak en az iki aday "
                          "kalmadı."),
                "ikili_isabet": ikili_isabet()}

    agg = copeland_borda(yarisan, zler)
    boot = bootstrap_sira(yarisan, zler, seed=seed)
    kademeler = kademele(yarisan, agg["bilesik"], boot)
    pay = boyutlandirma(adaylar, boot, vetolar)

    ikili = {}
    for i in range(len(yarisan)):
        for j in range(i + 1, len(yarisan)):
            a, b = yarisan[i], yarisan[j]
            k, d = ikili_karsilastir(zler[a], zler[b])
            # KALİBRE getiri üstünlük olasılığı — asıl sorulan soru
            _za, _zb = agg["bilesik"].get(a), agg["bilesik"].get(b)
            _dz = ((_za - _zb) if (_za is not None and _zb is not None)
                   else None)
            _p = getiri_ustunluk_olasiligi(abs(_dz)) if _dz is not None else None
            ikili[f"{a} vs {b}"] = {
                "sayac": k, "detay": d, "z_farki": (round(_dz, 2)
                                                    if _dz is not None else None),
                "onde": (a if (_dz or 0) > 0 else b) if _dz else None,
                "p_getiri_ustunlugu": (round(_p, 3) if _p else None)}

    isabet = ikili_isabet()
    return {
        "adaylar": adaylar, "yarisan": yarisan, "vetolar": vetolar,
        "z": zler, "kapsam": kapsam, "bilesik": agg["bilesik"],
        "borda": agg["borda"], "copeland": agg["copeland"],
        "siralama_saglam": agg["anlasiyor"], "bootstrap": boot,
        "kademeler": kademeler, "ikili": ikili, "boyutlandirma": pay,
        "ikili_isabet": isabet,
        "referans_kaynak": ("canlı örneklem" if referans
                            else "gömülü yaklaşık dağılımlar"),
        "durustluk": (
            f"Bu sıralama ~%{isabet*100:.0f} ikili isabet bekletir — yani "
            f"yazı-turadan biraz iyidir, KEHANET DEĞİLDİR. 3-4 aylık ufukta "
            f"bir hissenin hareketinin %65-85'i firmaya özgüdür ve "
            f"öngörülemez (Campbell-Lettau-Malkiel-Xu 2001). Aynı kademedeki "
            f"isimler İSTATİSTİKSEL OLARAK AYIRT EDİLEMEZ."),
        "kapsam_disi": (
            "Değer (DCF, F/K), kalite (Piotroski, ROIC) ve PEAD bu "
            "SIRALAMAYA GİRMEZ — 3-4 aylık ufukta ayrıştırıcı değiller "
            "(değer yakınsaması ~34 ayda söner: Bartram-Grinblatt; "
            "büyük-cap'te PEAD 2006'dan beri yok: Martineau 2022). "
            "Onlar VETO ve BAĞLAM katmanında duruyor — silinmediler, "
            "doğru yere konuldular."),
    }


MOD_TEK = "🔬 Tek Hisse Analizi"
MOD_TARAMA = "🎯 Alım Bölgesine Yaklaşanlar"
MOD_RAKIP = "⚔️ Rakip Kıyası"
MOD_PORTFOY = "🧺 Portföy Riski"
MOD_PIYASA = "📉 Piyasa Bağlamı"
MOD_KRIPTO = "🪙 Kripto Bağlamı"

_SP100 = ("AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK "
          "BKNG BLK BMY BRK-B C CAT CHTR CL CMCSA COF COP COST CRM CSCO CVS "
          "CVX DE DHR DIS DOW DUK EMR EXC F FDX GD GE GILD GM GOOG GOOGL GS "
          "HD HON IBM INTC INTU ISRG JNJ JPM KHC KO LIN LLY LMT LOW MA MCD "
          "MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NVDA ORCL PEP "
          "PFE PG PLTR PM PYPL QCOM RTX SBUX SCHW SO SPG T TGT TMO TMUS TSLA "
          "TXN UNH UNP UPS USB V VZ WFC WMT XOM")
_NDX = ("AAPL ABNB ADBE ADI ADP ADSK AEP AMAT AMD AMGN AMZN ANSS APP ARM "
        "ASML AVGO AXON AZN BIIB BKNG BKR CCEP CDNS CDW CEG CHTR CMCSA COST "
        "CPRT CRWD CSCO CSGP CSX CTAS CTSH DASH DDOG DXCM EA EXC FANG FAST "
        "FTNT GEHC GFS GILD GOOG GOOGL HON IDXX ILMN INTC INTU ISRG KDP KHC "
        "KLAC LIN LRCX LULU MAR MCHP MDLZ MELI META MNST MRVL MSFT MSTR MU "
        "NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR PDD PEP PLTR PYPL QCOM "
        "REGN ROP ROST SBUX SNPS TEAM TMUS TSLA TTD TTWO TXN VRSK VRTX WBD "
        "WDAY XEL ZS")
_SP500 = ("A AAPL ABBV ABNB ABT ACGL ACN ADBE ADI ADM ADP ADSK AEE AEP AES "
 "AFL AIG AIZ AJG AKAM ALB ALGN ALL ALLE AMAT AMCR AMD AME AMGN AMP AMT "
 "AMZN ANET ANSS AON AOS APA APD APH APTV ARE ATO AVB AVGO AVY AWK AXON "
 "AXP AZO BA BAC BALL BAX BBY BDX BEN BF-B BG BIIB BK BKNG BKR BLDR BLK "
 "BMY BR BRK-B BRO BSX BX BXP C CAG CAH CARR CAT CB CBOE CBRE CCI CCL "
 "CDNS CDW CE CEG CF CFG CHD CHRW CHTR CI CINF CL CLX CMCSA CME CMG CMI "
 "CMS CNC CNP COF COO COP COR COST CPAY CPB CPRT CPT CRL CRM CRWD CSCO "
 "CSGP CSX CTAS CTRA CTSH CTVA CVS CVX CZR D DAL DAY DD DE DECK DFS DG "
 "DGX DHI DHR DIS DLR DLTR DOC DOV DOW DPZ DRI DTE DUK DVA DVN DXCM EA "
 "EBAY ECL ED EFX EG EIX EL ELV EMN EMR ENPH EOG EPAM EQIX EQR EQT ERIE "
 "ES ESS ETN ETR EVRG EW EXC EXPD EXPE EXR F FANG FAST FCX FDS FDX FE "
 "FFIV FI FICO FIS FITB FMC FOX FOXA FRT FSLR FTNT FTV GD GDDY GE GEHC "
 "GEN GEV GILD GIS GL GLW GM GNRC GOOG GOOGL GPC GPN GRMN GS GWW HAL HAS "
 "HBAN HCA HD HES HIG HII HLT HOLX HON HPE HPQ HRL HSIC HST HSY HUBB HUM "
 "HWM IBM ICE IDXX IEX IFF INCY INTC INTU INVH IP IPG IQV IR IRM ISRG IT "
 "ITW IVZ J JBHT JBL JCI JKHY JNJ JNPR JPM K KDP KEY KEYS KHC KIM KKR "
 "KLAC KMB KMI KMX KO KR KVUE L LDOS LEN LH LHX LII LIN LKQ LLY LMT LNT "
 "LOW LRCX LULU LUV LVS LW LYB LYV MA MAA MAR MAS MCD MCHP MCK MCO MDLZ "
 "MDT MET META MGM MHK MKC MKTX MLM MMC MMM MNST MO MOH MOS MPC MPWR MRK "
 "MRNA MS MSCI MSFT MSI MTB MTCH MTD MU NCLH NDAQ NDSN NEE NEM NFLX NI "
 "NKE NOC NOW NRG NSC NTAP NTRS NUE NVDA NVR NWS NWSA NXPI O ODFL OKE "
 "OMC ON ORCL ORLY OTIS OXY PANW PARA PAYC PAYX PCAR PCG PEG PEP PFE PFG "
 "PG PGR PH PHM PKG PLD PLTR PM PNC PNR PNW PODD POOL PPG PPL PRU PSA "
 "PSX PTC PWR PYPL QCOM RCL REG REGN RF RJF RL RMD ROK ROL ROP ROST RSG "
 "RTX RVTY SBAC SBUX SCHW SHW SJM SLB SMCI SNA SNPS SO SOLV SPG SPGI SRE "
 "STE STLD STT STX STZ SWK SWKS SYF SYK SYY T TAP TDG TDY TECH TEL TER "
 "TFC TGT TJX TMO TMUS TPL TPR TRGP TRMB TROW TRV TSCO TSLA TSN TT TTWO "
 "TXN TXT TYL UAL UBER UDR UHS ULTA UNH UNP UPS URI USB V VICI VLO VLTO "
 "VMC VRSK VRSN VRTX VST VTR VTRS VZ WAB WAT WBA WBD WDC WEC WELL WFC WM "
 "WMB WMT WRB WST WTW WY WYNN XEL XOM XYL YUM ZBH ZBRA ZTS")
# ── B12-2 (KRİTİK) ────────────────────────────────────────────────────
# Listelerin TARİH DAMGASI YOKTU — bayatlığı ölçmek bile mümkün değildi.
# S&P 500 bileşimi yılda ~20-25 kez değişir (Goldman: 5 yılda %20, 10
# yılda %36 devir); değişimlerin ~%90'ı çeyreklik dengelemede DEĞİL,
# birleşme/bölünme/iflas nedeniyle ARA DÖNEMDE olur. 1-2 yıllık bir
# listede beklenen bayat isim sayısı ~20-45 (%4-9).
# ÖLÇÜM (elle örnekleme): bu listede PARA (2024 birleşme), WBA (2024
# halka kapandı), SMCI/MRNA (NDX'ten 2024 çıkarıldı), ILMN (NDX'te hâlâ
# duruyor) gibi isimler bulundu.
# Bayat liste İLERİYE dönük bir taramada "hayatta kalma yanlılığı"
# yaratmaz (bugün var olan hisselerle işlem yapılır) AMA KAPSAMA açığı
# yaratır: yeni eklenenler (çoğu zaman en hızlı büyüyenler) listede yok.
EVREN_TARIHI = "2025-06"        # listelerin derlendiği yaklaşık dönem
EVREN_NOTU = (
    f"Evren listeleri GÖMÜLÜDÜR ve yaklaşık {EVREN_TARIHI} bileşimidir. "
    f"S&P 500 yılda ~20-25 bileşen değiştirir; bu liste muhtemelen ~20-45 "
    f"bayat isim içeriyor ve son eklenenleri KAPSAMIYOR. Veri dönmeyen "
    f"semboller atlanır — atlanan sayısı yüksekse liste eskimiştir. "
    f"Kesinlik için 'Özel liste' kullan ya da güncel bileşeni endeks "
    f"sağlayıcısından/ETF portföy dosyasından al.")
EVRENLER = {
    "ABD": {"S&P 100 (hızlı, ~100)": tuple(_SP100.split()),
            "Nasdaq-100 (~100)": tuple(_NDX.split()),
            "S&P 500 (~500 — uzun sürer)": tuple(dict.fromkeys(
                _SP500.split()))},
}


def bolge_durumu(pahal):
    """Pahalılık skorundan alım bölgesi durumu. Bölge = skor < 35 (CAZİP)."""
    if pahal < 35:
        return "🎯 BÖLGEDE", 0
    if pahal < 45:
        return "🔥 ÇOK YAKIN", 1
    if pahal < 55:
        return "⏳ YAKLAŞIYOR", 2
    return "🧊 UZAK", 3


def tarama_sirala(rows, olcut="yakinlik"):
    """Yaklaşanlar: önce BÖLGEDE, sonra bölgeye mesafe artan; eşitlikte
    kalite yüksek önde. Alternatif: genel seçim puanı."""
    if olcut == "secim":
        return sorted(rows, key=lambda r: r["Seçim"], reverse=True)
    if olcut == "analist":
        return sorted(rows,
                      key=lambda r: (r.get("AnalistPot")
                                     if r.get("AnalistPot") is not None
                                     else -9.0),
                      reverse=True)
    return sorted(rows, key=lambda r: (r["_kademe"], r["Mesafe"],
                                       -r["Kalite"]))


def rakip_ozet(veriler):
    """
    RAKİP KIYASI motoru (saf, test edilebilir). veriler: [{t, isim, cur,
    sk(hizli_skorlar), }...]. Döner: metrik matrisi + medyan + rozetler
    (her metrikte en iyi) + uyarılar. Formatlama UI'da yapılır.
    """
    if not veriler or len(veriler) < 2:
        return None
    sira = [v["t"] for v in veriler]
    # (ad, yol, yön: 'min' iyi / 'max' iyi / None=nötr)
    tanim = [("Kalite", ("sk", "kalite"), "max"),
             ("Pahalılık", ("sk", "pahalilik"), "min"),
             ("Seçim Puanı", ("sk", "secim"), "max"),
             ("F/K (fwd)", ("met", "pe_f"), "min"),
             ("F/K (trl)", ("met", "pe_t"), "min"),
             ("EV/EBITDA", ("met", "ev_ebitda"), "min"),
             ("F/S", ("met", "ps"), "min"),
             ("P/FCF", ("met", "p_fcf"), "min"),
             ("PEG", ("met", "peg"), "min"),
             ("P/B", ("met", "pb"), None),
             ("Brüt Marj", ("met", "brut_marj"), "max"),
             ("Faaliyet Marjı", ("met", "faaliyet_marj"), "max"),
             ("Net Marj", ("met", "kar_marji"), "max"),
             ("ROE", ("met", "roe"), "max"),
             ("Gelir Büyümesi", ("met", "buyume"), "max"),
             ("NB/EBITDA", ("met", "nd_ebitda"), "min"),
             ("Cari Oran", ("met", "cari"), None),
             ("52h Konum", ("met", "poz52"), None),
             ("Analist Potansiyeli", ("met", "analist_pot"), "max"),
             ("Analist Tavsiye (1-5)", ("met", "tavsiye"), "min"),
             ("FCF Marjı", ("met", "fcf_marj"), "max"),
             ("200G Trend", ("met", "trend200"), None),
             ("52h Değişim", ("met", "chg52"), None),
             ("Beta", ("met", "beta"), None),
             ("Açığa Satış %Float", ("met", "short_f"), None),
             ("Fiyat", ("met", "fiyat"), None),
             ("Piyasa Değeri", ("met", "mcap"), None)]

    def oku(v, yol):
        kok = v["sk"] if yol[0] == "sk" else v["sk"]["metrikler"]
        return kok.get(yol[1])

    metrikler, medyan, rozetler = {}, {}, {t: [] for t in sira}
    for ad, yol, yon in tanim:
        satir = {v["t"]: oku(v, yol) for v in veriler}
        metrikler[ad] = satir
        say = sorted(x for x in satir.values() if x is not None)
        medyan[ad] = _medyan(say)             # BULGU U
        if yon and len(say) >= 2:
            hedef = min(say) if yon == "min" else max(say)
            for t, x in satir.items():
                if x is not None and x == hedef:
                    rozetler[t].append(("en düşük " if yon == "min"
                                        else "en yüksek ") + ad)
    # ── GRUP İÇİ YÜZDELİK SIRALAMA ───────────────────────────────────────
    # Mutlak bantların (ucuz/pahalı) YANINA göreli konum: her sütunda grup
    # içi sıra 0-100'e ölçeklenir, 4 eksende ağırlıklı toplanır. Eksik veri
    # o sütundan puan almaz (cezalandırılmaz) — dürüst boşluk.
    EKSEN = (("Değerleme", 0.30, (("F/K (fwd)", True), ("EV/EBITDA", True),
                                  ("F/S", True), ("P/FCF", True),
                                  ("PEG", True),
                                  ("Analist Potansiyeli", False))),
             ("Kalite", 0.30, (("Brüt Marj", False),
                               ("Faaliyet Marjı", False),
                               ("Net Marj", False), ("ROE", False),
                               ("FCF Marjı", False), ("NB/EBITDA", True))),
             ("Büyüme", 0.15, (("Gelir Büyümesi", False),)),
             ("Momentum", 0.25, (("200G Trend", False),
                                 ("52h Değişim", False),
                                 ("Analist Tavsiye (1-5)", True))))

    def _yuzdelik(ad, min_iyi):
        # ── B12-5 ────────────────────────────────────────────────────
        # N=3'te olası yüzdelikler yalnız 0 / 50 / 100'dür; "67/100" gibi
        # bir sayı, verinin taşıyamayacağı bir çözünürlük iddia eder.
        # Sayı KORUNDU (eksen toplama için gerekli) ama yanına SIRA
        # bilgisi konur ve arayüz sırayı gösterir.
        satir = [(t, x) for t, x in (metrikler.get(ad) or {}).items()
                 if x is not None]
        if len(satir) < 2:
            return {}
        satir.sort(key=lambda p: p[1], reverse=not min_iyi)  # en iyi önce
        n = len(satir)
        return {t: round(100.0 * (n - 1 - i) / (n - 1), 1)
                for i, (t, _) in enumerate(satir)}

    def _sira(ad, min_iyi):
        """B12-5: 'kaçıncı / kaç' — küçük N'de DÜRÜST gösterim."""
        satir = [(t, x) for t, x in (metrikler.get(ad) or {}).items()
                 if x is not None]
        if len(satir) < 2:
            return {}
        satir.sort(key=lambda p: p[1], reverse=not min_iyi)
        return {t: (i + 1, len(satir)) for i, (t, _) in enumerate(satir)}

    # ── B12-5b (derin motordaki B9-2 ile AYNI SINIF HATA) ────────────
    # Eksik metrik CEZASIZ atlanıyordu. Ama finansal veride eksiklik
    # RASTGELE DEĞİLDİR: F/K yoksa şirket zarar ediyordur, ROE yoksa
    # özsermaye negatiftir, P/FCF yoksa nakit akışı negatiftir. Yani
    # eksik olan metrikler TAM DA ALEYHTE olacak olanlardır; atmak
    # zayıf şirketi YUKARI çeker. Artık kapsama izlenir ve düşük
    # kapsama AÇIKÇA raporlanır.
    _pc = {}
    eksen_puan = {t: {} for t in sira}
    eksen_kapsama = {t: {} for t in sira}
    for ad_e, _w, cols in EKSEN:
        for t in sira:
            ps = []
            for c, min_iyi in cols:
                if c not in _pc:
                    _pc[c] = _yuzdelik(c, min_iyi)
                p = _pc[c].get(t)
                if p is not None:
                    ps.append(p)
            eksen_puan[t][ad_e] = (round(sum(ps) / len(ps), 1)
                                   if ps else None)
            eksen_kapsama[t][ad_e] = (len(ps), len(cols))
    genel_puan = {}
    for t in sira:
        parts = [(eksen_puan[t][ad_e], w) for ad_e, w, _ in EKSEN
                 if eksen_puan[t][ad_e] is not None]
        genel_puan[t] = (round(sum(p * w for p, w in parts)
                               / sum(w for _, w in parts), 1)
                         if parts else None)
    grup_sira = sorted([t for t in sira if genel_puan.get(t) is not None],
                       key=lambda t: genel_puan[t], reverse=True)
    gerekce = {}
    for t in sira:
        d_ = eksen_puan[t]
        guc = [k for k, v in d_.items() if v is not None and v >= 66]
        zayif = [k for k, v in d_.items() if v is not None and v <= 33]
        parc = []
        if guc:
            parc.append("grupta güçlü: " + ", ".join(guc))
        if zayif:
            parc.append("grupta zayıf: " + ", ".join(zayif))
        gerekce[t] = ("; ".join(parc) if parc
                      else "grup ortalamasına yakın profil")

    uyarilar = []
    for v in veriler:
        m = v["sk"]["metrikler"]
        if m.get("fcf") is not None and m["fcf"] <= 0:
            uyarilar.append(f"{v['t']}: FCF ≤ 0 — fiyat, henüz var olmayan "
                            f"nakde yaslanıyor.")
        nde = m.get("nd_ebitda")
        if nde is not None and nde > 4:
            uyarilar.append(f"{v['t']}: NB/EBITDA {nde}x — ağır borç.")
        if m.get("kar_marji") is not None and m["kar_marji"] < 0:
            uyarilar.append(f"{v['t']}: net marj negatif.")
    kaz = max(sira, key=lambda t: len(rozetler[t]))
    _siralar = {}
    for ad_e, _w, cols in EKSEN:
        for c, min_iyi in cols:
            if c not in _siralar:
                _siralar[c] = _sira(c, min_iyi)
    _N = len(sira)
    return {"sira": sira, "metrikler": metrikler, "medyan": medyan,
            "rozetler": rozetler, "uyarilar": uyarilar or None,
            "rozet_lideri": kaz,
            "eksen_puan": eksen_puan, "genel_puan": genel_puan,
            "eksen_kapsama": eksen_kapsama,
            "metrik_siralari": _siralar,
            "kucuk_n_uyarisi": (
                f"YALNIZ {_N} HİSSE KIYASLANIYOR. Bu boyutta '0-100 "
                f"yüzdelik' SAHTE KESİNLİKTİR: {_N} hissede olası "
                f"değerler yalnız "
                + ", ".join(f"{100*i/(_N-1):.0f}" for i in range(_N))
                + f". Sayı yerine SIRAYA bak ('2. / {_N}')."
                if _N < 10 else None),
            "eksik_metrik_notu": (
                "Eksik metrikler eksene KATILMAZ. Finansal veride "
                "eksiklik rastgele değildir (F/K yok = zarar, ROE yok = "
                "negatif özsermaye, P/FCF yok = negatif nakit akışı), "
                "yani eksik olan genellikle ALEYHTE olandır. Bu yüzden "
                "düşük kapsamalı bir eksen puanı OLDUĞUNDAN İYİ "
                "görünebilir — 'eksen_kapsama' alanına bak."),
            "peer_secim_uyarisi": (
                "Rakip listesini SEN seçtin. Kıyaslanabilir şirket "
                "seçimi, profesyonel değerlemede bilinen bir yanlılık "
                "kaynağıdır (gurur duyulan/ezilen rakip seçimi). "
                "Ölçüt önce belirlenmeli (sektör, ölçek, büyüme, "
                "kârlılık), isimler ondan SONRA gelmeli."),
            "grup_sira": grup_sira, "gerekce": gerekce,
            "siralama_notu": ("Yüzdelik puanlar GRUP İÇİ konumdur (100 = bu "
                              "grupta en iyi); mutlak ucuz/pahalı hükmü "
                              "DEĞİLDİR — pahalı bir grubun birincisi yine "
                              "pahalı olabilir; mutlak bant için Pahalılık "
                              "satırına bak. Ağırlık: Değerleme %30 · "
                              "Kalite %30 · Momentum %25 · Büyüme %15. "
                              "Momentum kısa ufuk (3-4 ay) bağlamıdır, yön "
                              "garantisi değildir.")}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def hizli_veri(t: str):
    """Tek sembol için HAFİF info seti (tarama). Hata → None, atlanır."""
    import yfinance as yf
    try:
        info, tk = {}, yf.Ticker(t, session=_yf_session())
        for g in ("get_info", "info"):
            try:
                o = getattr(tk, g)
                info = _yf_retry(lambda o=o: (o() if callable(o) else o),
                                 tries=2, base_wait=1.0)
                if info:
                    break
            except Exception:
                info = {}
        pr = _num(info.get("currentPrice")) or _num(
            info.get("regularMarketPrice"))
        if not info or pr is None:
            return None
        keys = ("marketCap", "trailingPE", "forwardPE",
                "priceToSalesTrailing12Months", "enterpriseToEbitda",
                "trailingPegRatio", "profitMargins", "grossMargins",
                "operatingMargins", "priceToBook", "returnOnEquity",
                "revenueGrowth", "totalDebt", "totalCash", "ebitda",
                "freeCashflow", "operatingCashflow", "debtToEquity",
                "currentRatio", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
                "currency", "sector",
                "targetMeanPrice", "recommendationMean", "beta",
                "totalRevenue", "fiftyDayAverage", "twoHundredDayAverage",
                "52WeekChange", "shortPercentOfFloat",
                # Denetim Bulgu 2: likidite + kazanç-takvimi vetolarının
                # girdileri (yfinance ADET döndürür; dolar = adet×fiyat)
                "averageVolume", "averageDailyVolume3Month",
                "averageDailyVolume10Day", "earningsTimestamp",
                "earningsTimestampStart")
        d = {k: info.get(k) for k in keys}
        d["fiyat"] = pr
        d["isim"] = (info.get("shortName") or info.get("longName") or t)[:26]
        return d
    except Exception:
        return None


def hizli_skorlar(d):
    """
    Hafif veriden hızlı KALİTE ve PAHALILIK (0-100) — tam motorun sadeleşmiş
    akrabası, aynı felsefeyle (_ramp bantları, uç kıstırma). Yalnızca ÖN
    ELEME içindir; nihai hüküm derin analizde verilir.
    """
    g = lambda k: _num(d.get(k))
    fiyat, mcap = g("fiyat"), g("marketCap")
    fcf, ocf = g("freeCashflow"), g("operatingCashflow")
    ebitda = g("ebitda")
    # DENETİM BULGUSU E: eski kod `(totalDebt or 0)` yazıyordu — borç verisi
    # GELMEDİĞİNDE borç 0 varsayılıyor, nakit tek başına "net nakit" gibi
    # okunuyor ve şirket +8 puan ÖDÜL alıyordu. Ana motor kendi kodunda tam
    # tersini söylüyor: "borç verisi YOKKEN 'borç = 0' varsaymak iyimser bir
    # uydurmadır" → net borç None bırakılır. Tarayıcı bu ilkeyi çiğniyordu.
    # DÜZELTME: borç yoksa nde=None → ne ödül ne ceza. Yan fayda: tarama_ele
    # içindeki "eksik_ele" filtresi (NB/EBITDA verisi yok) ÖLÜ KODDU, artık
    # gerçekten tetiklenebiliyor.
    _borc = g("totalDebt")
    nd = (_borc - (g("totalCash") or 0)) if _borc is not None else None
    nde = (nd / ebitda) if (nd is not None and ebitda and ebitda > 0) else None
    pm, roe, rg = g("profitMargins"), g("returnOnEquity"), g("revenueGrowth")
    de, cr = g("debtToEquity"), g("currentRatio")

    # ── Denetim Bulgu 6c: 'ayrisma_uyarisi' tarayıcının sektör-göreli
    # çarpanları "GÖREMEDİĞİNİ" söylüyordu — oysa sector alanı çekiliyor
    # ve medyan tabloları modül sabiti; görememe değil KULLANMAMAYDI.
    # Ana motorun B9-1 (KRİTİK) düzeltmesiyle aynı %70 göreli + %30
    # mutlak harman artık burada da uygulanır.
    def _gorelilestir(deger, anahtar, mutlak_puan):
        _med = ((SEKTOR_CARPAN_MEDYAN.get(d.get("sector") or "") or {})
                .get(anahtar) or GENEL_CARPAN_MEDYAN.get(anahtar))
        if not _med or _med <= 0 or mutlak_puan is None or not deger:
            return mutlak_puan
        _gor = _ramp(deger / _med, [(0.4, 12), (0.7, 30), (1.0, 50),
                                    (1.5, 68), (2.0, 80), (3.0, 92)])
        return 0.7 * _gor + 0.3 * mutlak_puan

    k = 15.0
    k += 15 if (fcf and fcf > 0) else (-20 if fcf is not None else -6)
    k += _ramp(pm, [(0, 0), (0.05, 8), (0.12, 14), (0.22, 20)]) or 0
    r_ = _ramp(roe, [(0, 0), (0.08, 8), (0.15, 14), (0.28, 20)]) or 0
    if de is not None and de > 200:
        r_ -= 8
    elif de is not None and de > 120:
        r_ -= 4
    k += max(r_, -8)
    k += _ramp(rg, [(-0.05, 0), (0.03, 5), (0.10, 9), (0.20, 14)]) or 0
    if nde is not None:
        k += 8 if nde < 1 else 4 if nde < 2.5 else 0 if nde < 4 else -10
    elif ebitda is not None and ebitda <= 0:
        k -= 8
    if cr is not None:
        k += 4 if cr >= 1.2 else (-5 if cr < 1 else 0)
    if ocf is not None and ocf > 0:
        k += 4
    kalite = float(max(0, min(100, k)))

    comps = []
    # ARAŞTIRMA TURU A4: trailing birincil (yfinance forwardPE bilinen
    # sorunlu; Damodaran medyanları trailing; analist tahmini iyimser).
    pe = g("trailingPE") or g("forwardPE")
    if pe and pe > 0:
        comps.append((_gorelilestir(pe, "pe",
                      _ramp(pe, [(7, 18), (12, 33), (18, 47), (28, 62),
                                 (45, 80), (70, 92)])), 1.5))
    ee = g("enterpriseToEbitda")
    if ee and ee > 0:
        comps.append((_gorelilestir(ee, "ev_ebitda",
                      _ramp(ee, [(5, 20), (9, 35), (14, 50), (22, 65),
                                 (32, 80), (45, 92)])), 1.5))
    ps = g("priceToSalesTrailing12Months")
    if ps and ps > 0:                    # ps için sektör medyanı yok →
        comps.append((_ramp(ps, [(1, 25), (3, 45), (6, 60), (12, 78),
                                 (20, 90)]), 1.2))          # mutlak kalır
    if fcf and fcf > 0 and mcap:
        comps.append((_gorelilestir(mcap / fcf, "p_fcf",
                      _ramp(mcap / fcf, [(6, 15), (12, 32), (18, 45),
                                         (28, 62), (45, 80),
                                         (70, 93)])), 2.0))
    elif fcf is not None and fcf <= 0:
        comps.append((78.0, 1.5))       # negatif FCF: fiyat vaade yaslanıyor
    peg = g("trailingPegRatio")
    if peg and 0 < peg < 20:
        comps.append((_ramp(peg, [(0.6, 25), (1.0, 42), (1.8, 58),
                                  (2.8, 74), (4.0, 88)]), 1.0))
    lo, hi = g("fiftyTwoWeekLow"), g("fiftyTwoWeekHigh")  # yalnız poz52
    # ── 52h BİLEŞENİ KALDIRILDI (denetim Bulgu 6 + araştırma bonusu):
    # Bu bileşen zirveye yakınlığı "pahalı" sayıyordu — George-Hwang
    # (2004) 52h etkisinin YÖNÜNÜ tersine çevirmek demekti; güncel
    # replikasyonlarda etki 20 piyasanın 18'inde POZİTİF ve uzun vadede
    # tersine dönmüyor. Ana motor aynı bileşeni B6-1'de silmişti; iki
    # motor aynı sinyali zıt işaretle kullanıyordu. poz52 bağlam sütunu
    # olarak tabloda DURUYOR — skora karışmıyor.
    pahal = (float(max(3, min(97, sum(v * w for v, w in comps)
                              / sum(w for _, w in comps))))
             if comps else 62.0)
    # ══ B12-1 (KRİTİK — DOKTRİN ÇELİŞKİSİ) ═══════════════════════════
    # Derin motorun doktrini: "kalite, pahalılığın mazereti DEĞİLDİR"
    # (TELAFİ ETMEYEN / non-compensatory kural). Ama bu formül TELAFİ
    # EDİCİ (compensatory): yüksek kalite, yüksek pahalılığı satın alıyor.
    # ÖLÇÜM (5 profil): "Balon fiyat + kusursuz şirket" (K=95, P=90)
    # seçim 61.0 alıyor ve "Ucuz + çürük"ün (44.8) ÜSTÜNDE sıralanıyor.
    # Daha kötüsü "Çok pahalı + çok kaliteli" (K=90, P=78) 62.8 ile
    # "Ucuz + orta şirket"in (63.0) yanına oturuyor — derin motor
    # birincisine "ÇOK PAHALI — DUR", ikincisine "CAZİP" diyor.
    # ÇÖZÜM: pahalılık bir VETO EKSENİDİR. Eşiğin üstünde seçim puanı
    # SERT biçimde cezalandırılır (telafi edilemez); altında kalite
    # sıralamayı belirler. Böylece tarayıcı derin motorla AYNI doktrini
    # uygular.
    _P_VETO, _P_UYARI = 70.0, 55.0
    if pahal >= _P_VETO:
        # derin motor bu bölgede "DUR" diyor → seçim puanı da öyle demeli
        secim = round(min(kalite * 0.6 + (100 - pahal) * 0.4,
                          100 - pahal), 1)
    elif pahal >= _P_UYARI:
        _ceza = (pahal - _P_UYARI) / (_P_VETO - _P_UYARI) * 15.0
        secim = round(kalite * 0.6 + (100 - pahal) * 0.4 - _ceza, 1)
    else:
        secim = round(kalite * 0.6 + (100 - pahal) * 0.4, 1)
    secim = float(max(0.0, min(100.0, secim)))
    met = {"fiyat": fiyat, "mcap": mcap, "pe_f": g("forwardPE"),
           "pe_t": g("trailingPE"), "ev_ebitda": ee, "ps": ps,
           "pb": g("priceToBook"), "peg": peg,
           "nd_ebitda": round(nde, 2) if nde is not None else None,
           "roe": roe, "buyume": rg, "fcf": fcf, "kar_marji": pm,
           "brut_marj": g("grossMargins"),
           "faaliyet_marj": g("operatingMargins"),
           "cari": cr,
           "poz52": (round((fiyat - lo) / (hi - lo), 2)
                     if (lo and hi and fiyat and hi > lo) else None),
           "p_fcf": (round(mcap / fcf, 1)
                     if (fcf and fcf > 0 and mcap) else None)}
    # Kıyas/tarama zenginleştirmesi: analist görünümü + trend + risk bağlamı.
    tm = g("targetMeanPrice")
    met["analist_pot"] = (round(tm / fiyat - 1, 3)
                          if (tm and fiyat) else None)
    met["tavsiye"] = g("recommendationMean")
    met["beta"] = g("beta")
    ma200 = g("twoHundredDayAverage")
    met["trend200"] = (round(fiyat / ma200 - 1, 3)
                       if (ma200 and fiyat) else None)
    met["chg52"] = g("52WeekChange")
    met["short_f"] = g("shortPercentOfFloat")
    rev_top = g("totalRevenue")
    met["fcf_marj"] = (round(fcf / rev_top, 3)
                       if (fcf is not None and rev_top) else None)
    return {"kalite": round(kalite, 1), "pahalilik": round(pahal, 1),
            "secim": secim, "metrikler": met,
            "secim_notu": (
                "Seçim puanı TELAFİ ETMEYEN kuralla hesaplanır: pahalılık "
                f"{_P_VETO:.0f} üstündeyse kalite bunu SATIN ALAMAZ "
                "(derin motorun doktriniyle aynı). Sıralama aracıdır, "
                "hüküm değildir."),
            "motor": "hızlı tarayıcı (hafif Yahoo info)",
            "ayrisma_uyarisi": (
                "Bu skor DERİN MOTORUN skoru DEĞİLDİR. Tarayıcı Piotroski, "
                "Altman, ROIC, tahakkuk kalitesi ve ters-DCF'i GÖREMEZ; "
                "sektör-göreli çarpan düzeltmesi ise artık burada da "
                "uygulanıyor (%70 göreli + %30 mutlak — ana motorla aynı). "
                "Ölçüm: iki motor pahalılık HÜKÜM KADEMESİNDE vakaların "
                "~%36'sında ayrışıyordu (korelasyon 0.82; ESKİ "
                "yapılandırmayla — 52h bileşeni kaldırılıp sektör-göreli "
                "eklendikten sonra YENİDEN ÖLÇÜLMEDİ, ayrışma azalmış "
                "olmalı ama sayı doğrulanmadı). Bu bir hata değil, farklı "
                "bilgi kümesidir — tarayıcı sıralaması derin hükmü "
                "ÖNCEDEN HABER VERMEZ.")}


def tarama_ele(sk, f):
    """Veto filtreleri: (geçti, elenme_sebebi)."""
    m = sk["metrikler"]
    if f["fcf_pozitif"] and not (m["fcf"] and m["fcf"] > 0):
        return False, ("FCF verisi yok" if m["fcf"] is None else "FCF ≤ 0")
    if f["marj_pozitif"] and not (m["kar_marji"] and m["kar_marji"] > 0):
        return False, ("marj verisi yok" if m["kar_marji"] is None else "kâr marjı ≤ 0")
    if m["mcap"] is None or m["mcap"] < f["min_mcap"]:
        return False, "piyasa değeri eşiği"
    nde = m["nd_ebitda"]
    if nde is None:
        if f["eksik_ele"]:
            return False, "NB/EBITDA verisi yok"
    elif nde > f["max_nde"]:
        return False, f"NB/EBITDA {nde}x"
    if sk["kalite"] < f["min_kalite"]:
        return False, "kalite eşiği"
    if sk["pahalilik"] > f["max_pahal"]:
        return False, "pahalılık eşiği"
    if f.get("trend_zorunlu"):
        tr = m.get("trend200")
        if tr is None or tr < 0:
            return False, "200G ortalama altı (trend)"
    mb = f.get("max_beta", 3.0)
    if mb < 3.0:
        be = m.get("beta")
        if be is not None and be > mb:
            return False, f"beta {be:.1f} > {mb:.1f}"
    return True, None


def tarama_neden(m):
    p = []
    if m.get("roe"):
        p.append(f"ROE %{m['roe']*100:.0f}")
    if m.get("p_fcf"):
        p.append(f"P/FCF {m['p_fcf']}x")
    if m.get("nd_ebitda") is not None:
        p.append(f"NB/EBITDA {m['nd_ebitda']}x")
    if m.get("pe_f"):
        p.append(f"fwd F/K {m['pe_f']:.0f}")
    if m.get("buyume") is not None:
        p.append(f"büyüme %{m['buyume']*100:.0f}")
    if m.get("trend200") is not None:
        p.append(f"200G %{m['trend200']*100:+.0f}")
    if m.get("analist_pot") is not None:
        p.append(f"analist %{m['analist_pot']*100:+.0f}")
    return " · ".join(p[:5])



# ═══════════════════ TÜM ABD ARTIMLI TARAMA (TA) ═══════════════════
# Araştırma raporu (screener yeniden tasarımı) uygulaması — Faz 1-3:
# nasdaqtrader evreni → 3 aşamalı huni → sqlite checkpoint/devam →
# kalite KAPISI → "yaklaşan" sıralaması. Eski modlara dokunulmadı.

_TA_DB = "tarama_tum_abd.db"


@st.cache_data(ttl=86400, show_spinner="Tüm ABD evreni indiriliyor…")
def evren_tum_abd():
    # nasdaqlisted.txt + otherlisted.txt (günlük). Hata → (None, hata_str)
    import io as _io
    import requests            # FIX: NameError — modülde global import yok
    baslik = {"User-Agent": EDGAR_UA}
    kok = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    ayik = ("WARRANT", "RIGHT", " UNIT", "PREFERRED", "DEPOSITARY SH",
            "NOTES DUE", "DEBENTURE")
    semboller, at_etf, at_isim, borsa = [], 0, 0, {}
    try:
        for dosya, s_i, n_i, etf_i, test_i, fin_i, etiket in (
                ("nasdaqlisted.txt", 0, 1, 6, 3, 4, "NASDAQ"),
                ("otherlisted.txt", 0, 1, 4, 6, None, "NYSE/AMEX")):
            r = requests.get(kok + dosya, headers=baslik, timeout=30)
            r.raise_for_status()
            for sat in _io.StringIO(r.text).read().splitlines()[1:]:
                if sat.startswith("File Creation Time"):
                    continue                      # footer
                k = sat.split("|")
                if len(k) <= max(s_i, n_i, etf_i, test_i):
                    continue
                sym, ad = k[s_i].strip(), k[n_i].upper()
                if (not sym or "$" in sym or k[test_i] == "Y"):
                    continue
                if k[etf_i] == "Y":
                    at_etf += 1
                    continue
                if fin_i is not None and k[fin_i] not in ("N", ""):
                    continue                      # deficient/bankrupt
                if any(x in ad for x in ayik):
                    at_isim += 1
                    continue
                _sym = sym.replace(".", "-")
                semboller.append(_sym)
                borsa.setdefault(_sym, etiket)
        semboller = sorted(dict.fromkeys(semboller))
        meta = {"n": len(semboller), "etf_at": at_etf, "isim_at": at_isim,
                "zaman": datetime.now().strftime("%Y-%m-%d %H:%M"),
                # A3/borsa filtresi: sembol → "NASDAQ" | "NYSE/AMEX"
                "borsa": borsa}
        return semboller, meta
    except Exception as e:
        # A-2 bulgusu: (None,hata) dönüşü 24s CACHE'leniyordu — tek geçici
        # ağ hatası evreni gün boyu kilitliyordu. Exception cache'lenmez;
        # çağıran yakalar, sonraki denemede yeniden indirilir.
        raise RuntimeError(f"{type(e).__name__}: {e}")


def _ta_con():
    import sqlite3
    con = sqlite3.connect(_TA_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS t(
        symbol TEXT PRIMARY KEY, stage INT, ts REAL,
        fiyat REAL, dh REAL, mcap REAL,
        kalite REAL, pahalilik REAL, secim REAL,
        bolge TEXT, kad TEXT, elenme TEXT, isim TEXT, cur TEXT,
        poz52 REAL, trend200 REAL, beta REAL, apot REAL, notu TEXT,
        sektor TEXT, borsa TEXT, fskor REAL, fskor_max REAL,
        z2 REAL, z2band TEXT, mom121 REAL, cek52w REAL,
        betaix REAL, r2ix REAL, dilim1 REAL, mesafe REAL,
        a3ts REAL, a3not TEXT)""")
    _ta_migrasyon(con)   # eski .db dosyalarına yeni kolonlar
    return con


# ── A3 YARDIMCILARI ─────────────────────────────────────────────────
def _ta_migrasyon(con):
    """Eski tarama_tum_abd.db dosyalarına yeni A3 kolonlarını ekler.
    ALTER TABLE idempotent değil → mevcut kolonlar okunup eksik eklenir."""
    mevcut = {r[1] for r in con.execute("PRAGMA table_info(t)")}
    yeni = {"sektor": "TEXT", "borsa": "TEXT", "fskor": "REAL",
            "fskor_max": "REAL", "z2": "REAL", "z2band": "TEXT",
            "mom121": "REAL", "cek52w": "REAL", "betaix": "REAL",
            "r2ix": "REAL", "dilim1": "REAL", "mesafe": "REAL",
            "a3ts": "REAL", "a3not": "TEXT"}
    for k, tip in yeni.items():
        if k not in mevcut:
            con.execute(f"ALTER TABLE t ADD COLUMN {k} {tip}")
    con.commit()


def ekran_dilim1(fiyat, pahal):
    """Derin motordaki kademeli_plan İSKONTO MERDİVENİNİN ekran kopyası
    — böylece tarayıcının 'bölgeye mesafe'si derin analizle AYNI dille
    konuşur. DCF ve destek/direnç çıpaları BURADA YOK (onlar derin
    analizde eklenir; seviyeyi oynatabilir):
        pahal < 45  → ilk dilim ~ fiyat × 0.98        (cazip tamponu)
        45 – 75     → iskonto %3 → %10 (skorla orantılı)
        ≥ 75        → İZLEME EŞİĞİ %10 → %35 (tavan) — alım dilimi DEĞİL
    Döner: (dilim1_fiyati, iskonto_orani)."""
    try:
        if fiyat is None or pahal is None or fiyat <= 0:
            return None, None
        p = float(pahal)
        if p < 45:
            isk = 0.02
        elif p < 75:
            isk = 0.03 + (p - 45) / 30 * 0.07
        else:
            isk = min(0.10 + (p - 75) / 22 * 0.25, 0.35)
        return round(float(fiyat) * (1 - isk), 2), round(isk, 4)
    except Exception:
        return None, None


def _a3_deger(df, adaylar, kol=0):
    """yfinance tablo satırı okuyucu — isim varyantlarına dayanıklı.
    kol=0 en güncel dönem, kol=1 bir önceki..."""
    if df is None:
        return None
    try:
        alt = {str(i).strip().lower(): i for i in df.index}
        for ad in adaylar:
            i = alt.get(ad.lower())
            if i is None:
                continue
            sat = df.loc[i]
            if hasattr(sat, "iloc"):
                if getattr(sat, "ndim", 1) > 1:      # yinelenen satır adı
                    sat = sat.iloc[0]
                if kol >= len(sat):
                    return None
                v = sat.iloc[kol]
            else:
                v = sat
            v = float(v)
            return v if v == v else None
    except Exception:
        return None
    return None


def _a3_fskor(inc, bal, cf):
    """Piotroski F — TARAYICI sürümü: ham yfinance yıllık tablolarından,
    derin motorla aynı ilkelerle (B5-3: ROA ve devir DÖNEM BAŞI varlığa
    bölünür). Veri eksik kriter ATLANIR → (geçen, mevcut_kriter)."""
    g = _a3_deger
    NI = ("Net Income", "Net Income Common Stockholders",
          "Net Income Continuous Operations")
    TA = ("Total Assets",)
    CFO = ("Operating Cash Flow",
           "Cash Flow From Continuing Operating Activities",
           "Total Cash From Operating Activities")
    LTD = ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")
    CA, CL = ("Current Assets",), ("Current Liabilities",)
    SH = ("Ordinary Shares Number", "Share Issued")
    GP, REV = ("Gross Profit",), ("Total Revenue", "Operating Revenue")

    ni0, ni1 = g(inc, NI, 0), g(inc, NI, 1)
    ta0, ta1, ta2 = g(bal, TA, 0), g(bal, TA, 1), g(bal, TA, 2)
    cfo0 = g(cf, CFO, 0)
    ltd0, ltd1 = g(bal, LTD, 0), g(bal, LTD, 1)
    ca0, cl0 = g(bal, CA, 0), g(bal, CL, 0)
    ca1, cl1 = g(bal, CA, 1), g(bal, CL, 1)
    sh0, sh1 = g(bal, SH, 0), g(bal, SH, 1)
    gp0, gp1 = g(inc, GP, 0), g(inc, GP, 1)
    rv0, rv1 = g(inc, REV, 0), g(inc, REV, 1)

    gecen, mevcut = 0, 0

    def kriter(kosul):
        nonlocal gecen, mevcut
        if kosul is None:
            return
        mevcut += 1
        if kosul:
            gecen += 1

    roa0 = (ni0 / ta1) if (ni0 is not None and ta1) else None
    roa1 = (ni1 / ta2) if (ni1 is not None and ta2) else None
    kriter(roa0 > 0 if roa0 is not None else None)                 # 1 ROA+
    kriter(cfo0 > 0 if cfo0 is not None else None)                 # 2 CFO+
    kriter((roa0 > roa1) if (roa0 is not None and roa1 is not None)
           else None)                                              # 3 ΔROA
    kriter((cfo0 > ni0) if (cfo0 is not None and ni0 is not None)
           else None)                                              # 4 tahakkuk
    kal0 = (ltd0 / ta0) if (ltd0 is not None and ta0) else (
        0.0 if (ltd0 is None and ta0 and bal is not None) else None)
    kal1 = (ltd1 / ta1) if (ltd1 is not None and ta1) else (
        0.0 if (ltd1 is None and ta1 and bal is not None) else None)
    kriter((kal0 <= kal1) if (kal0 is not None and kal1 is not None)
           else None)                                              # 5 kaldıraç↓
    cr0 = (ca0 / cl0) if (ca0 is not None and cl0) else None
    cr1 = (ca1 / cl1) if (ca1 is not None and cl1) else None
    kriter((cr0 > cr1) if (cr0 is not None and cr1 is not None)
           else None)                                              # 6 likidite↑
    kriter((sh0 <= sh1 * 1.02) if (sh0 and sh1) else None)         # 7 sulandırma yok
    gm0 = (gp0 / rv0) if (gp0 is not None and rv0) else None
    gm1 = (gp1 / rv1) if (gp1 is not None and rv1) else None
    kriter((gm0 > gm1) if (gm0 is not None and gm1 is not None)
           else None)                                              # 8 brüt marj↑
    dv0 = (rv0 / ta1) if (rv0 is not None and ta1) else None
    dv1 = (rv1 / ta2) if (rv1 is not None and ta2) else None
    kriter((dv0 > dv1) if (dv0 is not None and dv1 is not None)
           else None)                                              # 9 devir↑
    return (gecen, mevcut) if mevcut else (None, 0)


def _a3_altman(bal, inc, mcap, sektor):
    """Altman Z'' (üretim-dışı) — TARAYICI sürümü, derin motorla aynı
    doktrin: finansal sektörde HESAPLANMAZ (anlamsız); defter özsermaye
    ≤ 0 ise X4'te piyasa değerine düşülür ve etiketlenir.
        Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
        güvenli > 2.60 · gri 1.10–2.60 · tehlike < 1.10
    Döner: (z, band, not) — hesaplanamazsa (None, sebep, None)."""
    g = _a3_deger
    if sektor and any(k in str(sektor).lower()
                      for k in ("financial", "finans", "bank")):
        return None, "uygulanmaz (finansal)", None
    if bal is not None:
        try:  # sektör bilinmiyorsa mevduat satırı = banka sinyali
            if (not sektor) and any("deposit" in str(i).lower()
                                    for i in bal.index):
                return None, "uygulanmaz (mevduat satırı → finansal)", None
        except Exception:
            pass
    ta = g(bal, ("Total Assets",))
    if not ta:
        return None, "veri yok", None
    ca = g(bal, ("Current Assets",))
    cl = g(bal, ("Current Liabilities",))
    re_ = g(bal, ("Retained Earnings",))
    ebit = g(inc, ("EBIT", "Operating Income"))
    tl = g(bal, ("Total Liabilities Net Minority Interest", "Total Liab"))
    bve = g(bal, ("Stockholders Equity", "Common Stock Equity",
                  "Total Equity Gross Minority Interest"))
    x1 = ((ca - cl) / ta) if (ca is not None and cl is not None) else None
    x2 = (re_ / ta) if re_ is not None else None
    x3 = (ebit / ta) if ebit is not None else None
    notu = None
    if tl and tl > 0:
        if bve is not None and bve > 0:
            x4 = bve / tl
        elif mcap:
            x4 = mcap / tl
            notu = ("X4'te defter özsermaye ≤0/yok → PİYASA DEĞERİ "
                    "kullanıldı (derin motorun 'hizmet_piyasa_ikamesi' "
                    "kuralı) — piyasa değeri şişkinse Z'' iyimser görünür.")
        else:
            x4 = None
    else:
        x4 = None
    parcalar = [p for p in (x1, x2, x3, x4) if p is not None]
    if len(parcalar) < 3:
        return None, "veri eksik", None
    z = (6.56 * (x1 or 0.0) + 3.26 * (x2 or 0.0)
         + 6.72 * (x3 or 0.0) + 1.05 * (x4 or 0.0))
    band = ("güvenli" if z > 2.60 else
            "gri" if z >= 1.10 else "tehlike")
    if len(parcalar) < 4:
        notu = ((notu + " " if notu else "")
                + "Bir bileşen eksik — Z'' YAKLAŞIKTIR.")
    return round(float(z), 2), band, notu


def _ta_yaz(con, sym, **al):
    al["symbol"], al["ts"] = sym, __import__("time").time()
    kolon = ",".join(al)
    con.execute(f"INSERT INTO t({kolon}) VALUES({','.join('?'*len(al))}) "
                f"ON CONFLICT(symbol) DO UPDATE SET " +
                ",".join(f"{k}=excluded.{k}" for k in al if k != "symbol"),
                list(al.values()))
    con.commit()


def _ta_bekleyen(con, evren, ttl0=86400, ttl2=7 * 86400):
    import time as _t
    su = _t.time()
    ks = {r[0]: r for r in con.execute(
        "SELECT symbol,stage,ts,elenme FROM t")}
    a0, a1, a2 = [], [], []
    for s in evren:
        r = ks.get(s)
        if r is None or (r[1] == 0 and su - r[2] > ttl0):
            a0.append(s)
        elif r[3]:
            continue                              # elenmiş
        elif r[1] == 1:
            a1.append(s)
        elif r[1] == 2:
            a2.append(s)          # A2 kuyruğu (taze/bayat fark etmez)
        elif r[1] >= 3 and su - r[2] > ttl2:
            a0.append(s)                          # derin bayat → tazele
    return a0, a1, a2


def _ta_a0(con, grup):
    # Aşama-0: toplu fiyat+hacim (tek istek/chunk) → likidite kapısı
    import yfinance as yf      # FIX: sessiz NameError tamiri
    try:
        d = yf.download(grup, period="3mo", interval="1d",
                        group_by="ticker", threads=True, progress=False,
                        auto_adjust=False)
    except Exception as e:
        return f"A0: {type(e).__name__}: {e}"
    for s in grup:
        try:
            df = d[s] if len(grup) > 1 else d
            c = df["Close"].dropna()
            v = df["Volume"].dropna()
            if c.empty:
                _ta_yaz(con, s, stage=0, elenme="veri yok (A0)")
                continue
            f = float(c.iloc[-1])
            _vm = v.tail(20).mean()
            # A-2 bulgusu: hacim NaN ise 'nan<2e6' False kalıp likiditesi
            # KANITSIZ sembol A1'e sızıyordu — kanıt yoksa muhafazakâr ele.
            if _vm is None or _vm != _vm:
                _ta_yaz(con, s, stage=0, fiyat=f,
                        elenme="hacim verisi yok (A0)")
                continue
            dh = f * float(_vm)
            if f < 1.0:
                _ta_yaz(con, s, stage=0, fiyat=f, dh=dh,
                        elenme="fiyat < 1$ (penny)")
            elif dh < 2e6:
                _ta_yaz(con, s, stage=0, fiyat=f, dh=dh,
                        elenme="dolar-hacim < 2M$/g")
            else:
                _ta_yaz(con, s, stage=1, fiyat=f, dh=dh, elenme=None)
        except Exception:
            _ta_yaz(con, s, stage=0, elenme="veri yok (A0)")
    return None


def _ta_a1(con, s, min_mcap):
    # Aşama-1: fast_info mcap kapısı (ucuz tekil istek)
    import yfinance as yf      # FIX: sessiz NameError tamiri
    try:
        mc = yf.Ticker(s).fast_info.get("market_cap")
    except Exception as e:
        if any(k in str(e) for k in ("429", "999", "curl")):
            raise
        mc = None
    if mc is not None and mc < max(min_mcap, 3e8):
        _ta_yaz(con, s, stage=1, mcap=mc, elenme="mcap küçük (A1)")
    else:
        _ta_yaz(con, s, stage=2, mcap=mc, elenme=None)


def _ta_a2(con, s, filtre, borsa=None):
    # Aşama-2: mevcut derin yol (hizli_veri→skor→veto→bölge)
    d = hizli_veri(s)
    if d is None:
        _ta_yaz(con, s, stage=3, elenme="veri yok (A2)", borsa=borsa)
        return
    sk = hizli_skorlar(d)
    f2 = dict(filtre)
    f2["min_kalite"] = 0          # kalite eşiği KAPI'ya devredildi
    gecti, sebep = tarama_ele(sk, f2)
    m_ = sk["metrikler"]
    if not gecti:
        _ta_yaz(con, s, stage=3, elenme=sebep, kalite=sk["kalite"],
                pahalilik=sk["pahalilik"], isim=d.get("isim"),
                sektor=d.get("sector"), borsa=borsa)
        return
    durum, kad = bolge_durumu(sk["pahalilik"])
    _ta_yaz(con, s, stage=3, elenme=None, isim=d.get("isim"),
            fiyat=m_.get("fiyat"), mcap=m_.get("mcap"),
            kalite=sk["kalite"], pahalilik=sk["pahalilik"],
            secim=sk["secim"], bolge=durum, kad=kad,
            poz52=m_.get("poz52"), trend200=m_.get("trend200"),
            beta=m_.get("beta"), apot=m_.get("analist_pot"),
            notu=tarama_neden(m_),
            sektor=d.get("sector"), borsa=borsa,
            cur=("₺" if d.get("currency") == "TRY" else "$"))


def _ta_a3(con, s, endeks_seri):
    """Aşama-3: A2'yi geçen adaya TEKER TEKER derin katman.
    Sembol başına ~4 istek: fiyat geçmişi (2y) + 3 yıllık tablo.
    Üretilen: Piotroski F, Altman Z'' bandı, 12-1 momentum, 52h çekilme,
    MA200 mesafesi, endeks beta/R² ve derin motorla aynı iskonto
    merdiveninden 1. DİLİM + BÖLGEYE MESAFE. Hız sınırı (429/999) →
    raise (dış döngü geri çekilir); diğer hatalar kısmi yazılır."""
    import time as _t
    import yfinance as yf      # FIX: sessiz NameError tamiri
    row = con.execute(
        "SELECT fiyat, pahalilik, mcap, sektor FROM t WHERE symbol=?",
        (s,)).fetchone()
    fiyat_db, pahal, mcap, sektor = row if row else (None, None, None, None)
    tk = yf.Ticker(s, session=_yf_session())

    hist = None
    try:
        hist = _yf_retry(lambda: tk.history(period="2y",
                                            auto_adjust=False),
                         tries=2, base_wait=1.0)
    except Exception as e:
        if any(k in str(e) for k in ("429", "999", "curl")):
            raise
    kap = None
    if hist is not None and not getattr(hist, "empty", True) \
            and "Close" in hist:
        kap = hist["Close"].dropna()
        if getattr(kap.index, "tz", None) is not None:
            kap.index = kap.index.tz_localize(None)

    def _tablo(ad):
        try:
            df = _yf_retry(lambda: getattr(tk, ad), tries=2, base_wait=1.0)
            return df if (df is not None and hasattr(df, "empty")
                          and not df.empty) else None
        except Exception as e:
            if any(k in str(e) for k in ("429", "999", "curl")):
                raise
            return None

    inc, bal, cf = _tablo("income_stmt"), _tablo("balance_sheet"), \
        _tablo("cashflow")

    fs, fs_max = _a3_fskor(inc, bal, cf)
    z2, z2band, z2not = _a3_altman(bal, inc, mcap, sektor)

    mom121 = cek52w = trend200 = betaix = r2ix = None
    fiyat = fiyat_db
    if kap is not None and len(kap) >= 60:
        fiyat = float(kap.iloc[-1])
        if len(kap) >= 253:
            mom121 = round(float(kap.iloc[-22] / kap.iloc[-253] - 1), 4)
        z52 = float(kap.iloc[-252:].max()) if len(kap) >= 252 \
            else float(kap.max())
        cek52w = round(1.0 - fiyat / z52, 4) if z52 else None
        if len(kap) >= 200:
            trend200 = round(
                fiyat / float(kap.rolling(200).mean().iloc[-1]) - 1, 4)
        if endeks_seri is not None:
            hx = hisse_endeks_iliskisi(kap, endeks_seri)
            if hx and not hx.get("yetersiz"):
                betaix, r2ix = hx["beta"], hx["r2"]

    dilim1, isk = ekran_dilim1(fiyat, pahal)
    parc = []
    if fs is not None:
        parc.append(f"F {fs}/{fs_max}")
    if z2 is not None:
        parc.append(f"Z'' {z2} {z2band}")
    elif z2band:
        parc.append(f"Z'' {z2band}")
    if mom121 is not None:
        parc.append(f"12-1 %{mom121*100:+.0f}")
    if betaix is not None:
        parc.append(f"β {betaix:.1f} R² %{r2ix*100:.0f}")
    if z2not:
        parc.append("⚠︎ " + z2not.split(" — ")[0])
    if kap is None and inc is None and bal is None:
        parc = ["A3 verisi alınamadı"]
    _ta_yaz(con, s, stage=3, a3ts=_t.time(),
            fskor=fs, fskor_max=(fs_max or None),
            z2=z2, z2band=z2band, mom121=mom121, cek52w=cek52w,
            betaix=betaix, r2ix=r2ix,
            trend200=(trend200 if trend200 is not None else None),
            fiyat=fiyat, dilim1=dilim1, mesafe=isk,
            a3not=(" · ".join(parc) if parc else None))


def kademe_kapisi(kalite, kademe):
    if kalite is None:
        return kademe == "Hepsi"
    if kademe.startswith("Çok"):
        return kalite >= 70
    if kademe.startswith("Orta"):
        return 40 <= kalite < 70
    if kademe.startswith("Düşük"):
        return kalite < 40
    return True


def render_tum_abd_tarama(evren, filtre, meta):
    import time as _t
    st.markdown("### 🇺🇸 Tüm ABD — Artımlı Derin Tarama (v2: A3 katmanı)")
    bmap = meta.get("borsa") or {}

    # ── Borsa seçimi (kullanıcı isteği: 'Nasdaq'ta kaç hisse varsa') ──
    borsa_sec = st.radio(
        "Borsa evreni", ["Yalnız NASDAQ", "NASDAQ + NYSE + AMEX"],
        horizontal=True, index=0,
        help="nasdaqlisted.txt = NASDAQ'ta kote olanlar; otherlisted.txt "
             "= NYSE/AMEX vb. Tarama her iki durumda da HER HİSSEYİ "
             "TEKER TEKER hesaplar — fark yalnız kapsamdır.")
    if borsa_sec.startswith("Yalnız"):
        if bmap:
            evren = [s for s in evren if bmap.get(s) == "NASDAQ"]
        else:
            st.caption("ℹ️ Borsa etiketi eski evren önbelleğinde yok — "
                       "etiketli filtre için önbellek (24s) yenilenince "
                       "devreye girer; şimdilik tüm ABD listesi kullanılıyor.")
    st.caption(f"Evren: **{len(evren)}** ortak hisse "
               f"({borsa_sec}; ETF/warrant/preferred ayıklandı; kaynak: "
               f"nasdaqtrader, {meta.get('zaman','?')}). İlerleme "
               f"**{_TA_DB}** dosyasında saklanır — durdur/kapat, sonra "
               f"kaldığın yerden devam et. Fiyat katmanı 1 günde, derin "
               f"katman 7 günde bayatlar.")
    st.caption(
        "🔬 **Huni:** A0 toplu fiyat/likidite → A1 piyasa değeri → "
        "A2 hafif skor + veto → **A3 (yeni): her adaya TEKER TEKER "
        "bilanço + fiyat geçmişi** → Piotroski F, Altman Z'' bandı, "
        "12-1 momentum, endeks β/R² ve derin motorla AYNI iskonto "
        "merdiveninden **1. dilim + bölgeye mesafe**. A3 skorları "
        "TARAYICI kalitesindedir — nihai hüküm yine derin analizde.")

    kademe = st.radio("Kalite kademesi (KAPI — ucuzluk kaliteyi TELAFİ "
                      "ETMEZ)", ["Çok kaliteli (≥70)", "Orta (40-69)",
                                 "Düşük (<40) ⚠️", "Hepsi"],
                      horizontal=True, index=0)
    if kademe.startswith("Düşük"):
        st.warning("Düşük kademe: temel sağlamlık kapısını GEÇEMEYEN "
                   "hisseler. Liste gözlem içindir; ucuzluk düşük "
                   "kaliteyi telafi etmez. Yatırım tavsiyesi değildir.")
    a3_kapsam = st.radio(
        "A3 derin katman kapsamı",
        ["🎯 Pahalılık < 45 (bölgede + çok yakın) — önerilen",
         "⏳ Pahalılık ≤ 55 (yaklaşanlar dahil)",
         "Hepsi (A2'yi geçen tüm hisseler — en uzun)"],
        horizontal=True, index=0,
        help="A3 sembol başına ~4 istek yapar (fiyat geçmişi + 3 tablo). "
             "Bölgeye uzak hisselerde bu maliyet erken harcanmış olur — "
             "istersen yine de 'Hepsi' ile evrenin tamamına uygula.")
    a3_esik = (45.0 if a3_kapsam.startswith("🎯")
               else 55.0 if a3_kapsam.startswith("⏳") else 1e9)

    con = _ta_con()
    a0, a1, a2 = _ta_bekleyen(con, evren)
    ev_set = set(evren)
    su = _t.time()
    a3 = [r[0] for r in con.execute(
        "SELECT symbol, pahalilik, a3ts FROM t WHERE stage>=3 AND "
        "elenme IS NULL AND pahalilik IS NOT NULL "
        "ORDER BY pahalilik ASC")
        if r[0] in ev_set and r[1] is not None and r[1] < a3_esik
        and (r[2] is None or su - r[2] > 7 * 86400)]
    n_el, n_ok, n_a3 = [con.execute(q).fetchone()[0] for q in (
        "SELECT COUNT(*) FROM t WHERE elenme IS NOT NULL",
        "SELECT COUNT(*) FROM t WHERE stage>=3 AND elenme IS NULL",
        "SELECT COUNT(*) FROM t WHERE a3ts IS NOT NULL AND elenme IS NULL")]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("A0 bekleyen", len(a0)); c2.metric("A1 bekleyen", len(a1))
    c3.metric("A2 bekleyen", len(a2)); c4.metric("A3 bekleyen", len(a3))
    c5.metric("Elenen", n_el); c6.metric("A2✓ / A3✓", f"{n_ok}/{n_a3}")
    kalan_derin = len(a2) + int(len(a1) * 0.35) + int(len(a0) * 0.12)
    st.caption(f"Kaba ETA: A0 ~{len(a0)//150 + 1} toplu istek · A2 kuyruk "
               f"~{kalan_derin} sembol (~{kalan_derin * 2.2 / 60:.0f} dk) "
               f"· A3 kuyruk {len(a3)} sembol (~"
               f"{len(a3) * 5.5 / 60:.0f} dk — sembol başına ~4 istek). "
               f"Uzun kuyrukta taramayı birkaç güne yay — checkpoint "
               f"bunun için var.")
    st.session_state.setdefault("_ta_calis", False)
    st.session_state.setdefault("_ta_gec", 0.5)
    st.session_state.setdefault("_ta_hata", 0)
    st.session_state.setdefault("_ta_a2say", 0)
    st.session_state.setdefault("_ta_a3say", 0)
    b1, b2 = st.columns(2)
    if b1.button("▶️ Başlat / Devam", type="primary", width="stretch",
                 disabled=not evren):
        st.session_state["_ta_calis"] = True
        st.session_state["_ta_hata"] = 0
        st.rerun()
    if b2.button("⏸️ Durdur", width="stretch"):
        st.session_state["_ta_calis"] = False
    if st.session_state["_ta_calis"]:
        try:
            if a0:
                grup = a0[:150]
                hata = _ta_a0(con, grup)
                if hata and any(k in hata for k in ("429", "999")):
                    raise RuntimeError(hata)
                st.toast(f"A0: {len(grup)} sembol işlendi")
            elif a1:
                for s in a1[:40]:
                    _ta_a1(con, s, filtre.get("min_mcap", 3e8))
                    _t.sleep(st.session_state["_ta_gec"] * 0.5)
                st.toast(f"A1: {min(40, len(a1))} sembol")
            elif a2:
                for s in a2[:15]:
                    st.session_state["_ta_a2say"] += 1
                    if st.session_state["_ta_a2say"] % 100 == 0:
                        _t.sleep(30)              # B12-6 proaktif mola
                    _ta_a2(con, s, filtre, bmap.get(s))
                    _t.sleep(st.session_state["_ta_gec"])
                st.toast(f"A2: {min(15, len(a2))} derin sembol")
            elif a3:
                _ix = fetch_endeks("^IXIC", "2y")   # önbellekli; β/R² için
                for s in a3[:6]:
                    st.session_state["_ta_a3say"] += 1
                    if st.session_state["_ta_a3say"] % 50 == 0:
                        _t.sleep(30)              # A3 proaktif mola
                    _ta_a3(con, s, _ix)
                    _t.sleep(st.session_state["_ta_gec"] * 1.5)
                st.toast(f"A3: {min(6, len(a3))} sembol derinleştirildi")
            else:
                st.session_state["_ta_calis"] = False
                st.success("✓ Tarama güncel — A0/A1/A2/A3 kuyruğu boş.")
            if st.session_state["_ta_calis"]:
                st.rerun()
        except Exception as e:
            st.session_state["_ta_hata"] += 1
            st.session_state["_ta_gec"] = min(
                st.session_state["_ta_gec"] * 1.7, 8.0)
            if st.session_state["_ta_hata"] >= 6:
                st.session_state["_ta_calis"] = False
                st.error(f"Yahoo hız sınırı görünüyor ({e}). 15-30 dk "
                         f"sonra '▶️ Devam' ile sürdür — ilerleme kayıtlı.")
            else:
                _t.sleep(3 * st.session_state["_ta_hata"])
                st.rerun()
    # ── sonuç tablosu (canlı) ──
    import pandas as _pd
    df = _pd.read_sql_query(
        "SELECT symbol, isim, borsa, fiyat, mcap, kalite, pahalilik, "
        "fskor, fskor_max, z2, z2band, mom121, trend200, betaix, r2ix, "
        "dilim1, mesafe, secim, a3ts, notu, a3not FROM t "
        "WHERE stage>=3 AND elenme IS NULL AND pahalilik IS NOT NULL",
        con)
    if df.empty:
        st.info("Henüz derin-tamam sonuç yok — taramayı başlat.")
        return
    if borsa_sec.startswith("Yalnız") and bmap:
        df = df[df["symbol"].map(lambda s: bmap.get(s) == "NASDAQ")
                | df["borsa"].eq("NASDAQ")]
    df = df[df["kalite"].apply(lambda k: kademe_kapisi(k, kademe))]
    if df.empty:
        st.info("Bu kademe kapısını geçen sonuç yok.")
        return
    # dilim/mesafe A3 beklemeden de hesaplanabilir (fiyat+pahalılıktan):
    _dm = df.apply(lambda r: ekran_dilim1(r["fiyat"], r["pahalilik"]),
                   axis=1, result_type="expand")
    df["dilim1"] = df["dilim1"].fillna(_dm[0])
    df["mesafe"] = df["mesafe"].fillna(_dm[1])
    df["_kad"] = df["pahalilik"].apply(lambda p: bolge_durumu(p)[1])
    df["Durum"] = df["pahalilik"].apply(lambda p: bolge_durumu(p)[0])
    z_gizle = st.checkbox("🚫 Z'' TEHLİKE bandını gizle (iflas riski "
                          "bandı — Altman Z'' < 1.10)", value=False)
    if z_gizle:
        df = df[df["z2band"].ne("tehlike") | df["z2band"].isna()]
    df = df.sort_values(
        ["_kad", "mesafe", "fskor", "kalite"],
        ascending=[True, True, False, False]).reset_index(drop=True)
    st.markdown(f"#### Sonuç — {kademe} kapısını geçen **{len(df)}** "
                f"hisse (bölge → mesafe → F-skor → kalite sırasıyla)")
    st.caption(
        "**1. Dilim** = derin motordaki kademeli_plan ile AYNI iskonto "
        "merdiveni (pahalılık<45 → ~%2 · 45-75 → %3-10 · ≥75 → izleme "
        "eşiği %10-35). DCF ve destek/direnç çıpası BURADA YOK — kesin "
        "seviyeler adayın derin analizindeki Kademeli Plan'dadır. "
        "**Mesafe** = fiyatın 1. dilime inmesi gereken yüzde. F ve Z'' "
        "tarayıcı kalitesindedir (ham yıllık tablolar); bantlar kalibre "
        "edilmemiş göreli konumdur. Top-liste SEÇİM ETKİSİ taşır. "
        "Tavsiye değildir.")
    gost = _pd.DataFrame({
        "Ticker": df["symbol"], "İsim": df["isim"],
        "Borsa": df["borsa"].fillna("?"),
        "Fiyat": df["fiyat"].round(2),
        "Kalite": df["kalite"], "Pahalılık": df["pahalilik"],
        "Durum": df["Durum"],
        "1.Dilim": df["dilim1"],
        "Mesafe": df["mesafe"].map(
            lambda m: f"%{m*100:.1f}" if m == m else "—"),
        "F": df.apply(lambda r: (f"{r['fskor']:.0f}/{r['fskor_max']:.0f}"
                                 if r["fskor"] == r["fskor"] else "…"),
                      axis=1),
        "Z''": df.apply(lambda r: (f"{r['z2']:.1f} {r['z2band']}"
                                   if r["z2"] == r["z2"]
                                   else (r["z2band"] or "…")), axis=1),
        "12-1": df["mom121"].map(
            lambda m: f"%{m*100:+.0f}" if m == m else "…"),
        "β/R²(IXIC)": df.apply(
            lambda r: (f"{r['betaix']:.1f} / %{r['r2ix']*100:.0f}"
                       if r["betaix"] == r["betaix"] else "…"), axis=1),
        "A3": df["a3ts"].map(lambda t_: "✓" if t_ == t_ else "…"),
        "Not": df["a3not"].fillna(df["notu"]).fillna(""),
    })
    st.dataframe(gost.head(40), hide_index=True, width="stretch")
    st.download_button("⬇️ Tam sonuç (CSV)",
                       gost.to_csv(index=False).encode("utf-8-sig"),
                       "tum_abd_tarama.csv", "text/csv")
    st.caption("… işaretli hücreler A3 kuyruğunda — '▶️ Devam' ile "
               "teker teker dolar. Bir adayı incelemek için sembolü "
               "kopyala → sol menüden Tek Hisse moduna gir. (Skorlar bu "
               "taramanın motorundandır; nihai karar tam analiz "
               "ekranındadır.)")
# ═════════════════ /TÜM ABD ARTIMLI TARAMA ═════════════════


def render_screener():
    st.markdown("# 🎯 Alım Bölgesine Yaklaşanlar")
    st.caption("Amaç 'en iyi şirket' listesi DEĞİL: kalite vetosunu geçen "
               "şirketlerden fiyatı bizim CAZİP bölgemize (pahalılık < 35) "
               "girmiş ya da yaklaşmakta olanları bulmak. Kesin giriş "
               "seviyeleri, adayın DERİN analizindeki Kademeli Plan'dadır. "
               "Evren listeleri gömülüdür (~2025, yaklaşık); veri gelmeyen "
               "semboller atlanır. Kesinlik istersen Özel Liste kullan.")
    pazar = "ABD"
    ev_ops = (list(EVRENLER[pazar].keys()) + ["Özel liste"]
              + ["🇺🇸 TÜM ABD (artımlı — uzun tarama)"])
    ev_sec = st.selectbox("Evren", ev_ops)
    if ev_sec == "Özel liste":
        ozel = st.text_area(
            "Sembolleri boşluk/virgülle yaz (ör: AAPL MSFT NVDA)", height=80)
        evren = tuple(dict.fromkeys(
            t.strip().upper() for t in ozel.replace(",", " ").split()
            if t.strip()))
    elif ev_sec.startswith("🇺🇸"):
        try:
            _ev, _meta = evren_tum_abd()
            evren = tuple(_ev)
        except Exception as _e:
            st.error(f"Evren indirilemedi ({_e}) — bağlantıyı kontrol "
                     f"edip yeniden dene; hata önbelleğe ALINMAZ, gömülü "
                     f"listeler etkilenmez.")
            evren, _meta = (), {}
    else:
        evren = EVRENLER[pazar][ev_sec]

    with st.expander("🎚️ Filtreler (veto kuralları)", expanded=True):
        f1, f2, f3 = st.columns(3)
        min_kalite = f1.slider("Kalite skoru ≥", 0, 90, 55,
                               help="Hızlı kalite skorunun alt sınırı.")
        max_pahal = f2.slider("Pahalılık skoru ≤", 10, 97, 65,
                              help="Yaklaşanları görmek için 65 önerilir; "
                                   "35 altı zaten BÖLGEDE sayılır.")
        max_nde = f3.slider("Net Borç/EBITDA ≤", 0.5, 6.0, 3.5, 0.5)
        f4, f5, f6 = st.columns(3)
        fcf_poz = f4.checkbox("FCF > 0 zorunlu", value=True,
                              help="ORCL dersi: negatif FCF'li şirket "
                                   "listeye giremez.")
        marj_poz = f5.checkbox("Kâr marjı > 0 zorunlu", value=True)
        eksik_ele = f6.checkbox("Borç verisi eksikse ele", value=False,
                                help="Kapalıyken eksik NB/EBITDA veto "
                                     "edilmez, sadece işaretlenir.")
        f7, f8 = st.columns(2)
        trend_zorunlu = f7.checkbox(
            "Fiyat 200 günlük ortalamanın ÜSTÜNDE olsun", value=False,
            help="Kısa ufukta (3-4 ay) 'düşen bıçak' yerine trend teyidi "
                 "arayanlar için BAĞLAM filtresi — sinyal/garanti değildir; "
                 "trend altı ucuzlayan hisse daha da ucuzlayabilir.")
        max_beta = f8.slider(
            "Maks. Beta (oynaklık)", 0.5, 3.0, 3.0, 0.1,
            help="3.0 = filtre kapalı. Kısa ufukta yüksek beta = geniş "
                 "dalgalanma bandı = dilimlerin/stopun daha sık test "
                 "edilmesi demektir.")
        min_mcap = st.number_input(
            "Min. piyasa değeri (USD)",
            min_value=0.0, value=2e9, step=1e9,
            format="%.0f")
    # B12-3: kaç farklı filtre ayarı denendiğini SAY (belirtim arama
    # ölçüsü). Deflated Sharpe Ratio da girdisi olarak deneme sayısını
    # alır — kullanıcı bu sayıyı görmeli.
    _imza = (min_kalite, max_pahal, max_nde, fcf_poz, marj_poz,
             eksik_ele, min_mcap, trend_zorunlu, max_beta)
    if st.session_state.get("_filtre_imza") != _imza:
        st.session_state["_filtre_imza"] = _imza
        st.session_state["_filtre_denemesi"] = (
            st.session_state.get("_filtre_denemesi", 0) + 1)
    filtre = {"min_kalite": min_kalite, "max_pahal": max_pahal,
              "max_nde": max_nde, "fcf_pozitif": fcf_poz,
              "marj_pozitif": marj_poz, "eksik_ele": eksik_ele,
              "min_mcap": min_mcap, "trend_zorunlu": trend_zorunlu,
              "max_beta": max_beta}
    if ev_sec.startswith("🇺🇸"):
        render_tum_abd_tarama(evren, filtre, _meta)
        return

    st.caption(f"Evren: **{len(evren)}** sembol · ilk tarama sembol başına "
               f"~0.5-1.5 sn sürer (önbellek 6 saat — tekrar tarama "
               f"hızlıdır).")
    if st.button("🔎 Taramayı Başlat", type="primary",
                 width="stretch", disabled=not evren):
        # Denetim Bulgu 5: tarama_ele SEBEP üretiyor ama '_' ile
        # atılıyordu — kullanıcı hangi hissenin NEDEN elendiğini
        # göremiyordu. Artık toplanıp gösteriliyor.
        sonuc, elenen, hatali, elenen_detay = [], 0, 0, []
        prog = st.progress(0.0, text="Başlıyor…")
        # B12-6: Yahoo'nun yayımlanmış hız sınırı YOK ama topluluk
        # deneyimi ~100 istekten sonra ~30 sn duraklama öneriyor; aşımda
        # IP saatlerce bloklanabiliyor. Tepkisel yeniden deneme (mevcut
        # _yf_retry) yetmez — PROAKTİF kısıtlama gerekir.
        import time as _t
        for i, t in enumerate(evren):
            if i and i % 100 == 0:
                prog.progress(i / len(evren),
                              text=f"hız sınırı koruması — 30 sn bekleniyor "
                                   f"({i}/{len(evren)})")
                _t.sleep(30)
            elif i and i % 10 == 0:
                _t.sleep(0.4)
            d = hizli_veri(t)
            if i % 3 == 0 or i == len(evren) - 1:
                prog.progress((i + 1) / len(evren),
                              text=f"{t} · {i+1}/{len(evren)}")
            if d is None:
                hatali += 1
                continue
            sk = hizli_skorlar(d)
            gecti, sebep = tarama_ele(sk, filtre)
            if not gecti:
                elenen += 1
                elenen_detay.append({"Ticker": t, "Sebep": sebep})
                continue
            m_ = sk["metrikler"]
            durum, kad = bolge_durumu(sk["pahalilik"])
            sonuc.append({"Ticker": t, "İsim": d["isim"],
                          "Fiyat": m_["fiyat"], "PD": m_["mcap"],
                          "Kalite": sk["kalite"],
                          "Pahalılık": sk["pahalilik"],
                          "Mesafe": round(sk["pahalilik"] - 35.0, 1),
                          "Durum": durum, "_kademe": kad,
                          "52h": m_.get("poz52"),
                          "Trend200": m_.get("trend200"),
                          "AnalistPot": m_.get("analist_pot"),
                          "Beta": m_.get("beta"),
                          "Seçim": sk["secim"],
                          "Not": tarama_neden(m_),
                          "_cur": ("₺" if d.get("currency") == "TRY"
                                   else "$")})
        prog.empty()
        sonuc = tarama_sirala(sonuc, "yakinlik")
        # B12-2: atlanan sembol oranı, listenin BAYATLIK göstergesidir.
        _bayat_oran = hatali / max(len(evren), 1)
        st.session_state["tarama"] = {
            "pazar": pazar, "top": sonuc[:20],
            "atlanan": hatali, "evren_boyu": len(evren),
            "elenen_detay": elenen_detay[:60],
            "bayat_oran": round(_bayat_oran, 3),
            "istatistik": (f"{len(evren)} sembol → {hatali} veri yok "
                           f"(%{_bayat_oran*100:.1f}) · "
                           f"{elenen} vetoyla elendi · "
                           f"{len(sonuc)} geçti → Top "
                           f"{min(20, len(sonuc))}"),
            "evren_tarihi": EVREN_TARIHI}
        try:
            gunluk_yaz({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "ticker": f"🔎TARAMA-{pazar}", "fiyat": None,
                        "skor": None, "kalite": None,
                        "hukum": ", ".join(r["Ticker"]
                                           for r in sonuc[:20])[:60]})
        except Exception:
            pass

    ts = st.session_state.get("tarama")
    if ts:
        st.markdown(f"### 🎯 Alım Bölgesine En Yakın "
                    f"{len(ts['top'])} — {ts['pazar']}")
        st.caption(ts["istatistik"])
        if ts.get("elenen_detay"):
            with st.expander(f"🚫 Vetoyla elenenler ve SEBEPLERİ "
                             f"({len(ts['elenen_detay'])}"
                             f"{'+' if len(ts['elenen_detay']) == 60 else ''})"):
                st.dataframe(pd.DataFrame(ts["elenen_detay"]),
                             hide_index=True, width="stretch")
                st.caption("Denetim düzeltmesi: elenme sebebi üretilip "
                           "atılıyordu — artık görünür. Sebep sütunu, "
                           "tarama_ele veto filtrelerinden gelir; eşik "
                           "gevşetince kimin geri geleceğini buradan gör.")
        if not ts["top"]:
            st.warning("Hiçbir sembol filtrelerden geçemedi — eşikleri "
                       "gevşetmeyi dene (ör. kalite ≥ 45, pahalılık ≤ 75).")
            return
        olcut = st.radio("Sıralama", ["🎯 Bölgeye yakınlık",
                                      "🏆 Genel seçim puanı",
                                      "📣 Analist potansiyeli"],
                         horizontal=True)
        top = tarama_sirala(list(ts["top"]),
                            "secim" if olcut.startswith("🏆")
                            else "analist" if olcut.startswith("📣")
                            else "yakinlik")
        bolgede = sum(1 for r in top if r["_kademe"] == 0)
        st.caption(f"🎯 Bölgede: **{bolgede}** · 🔥 Çok yakın: "
                   f"**{sum(1 for r in top if r['_kademe'] == 1)}** · "
                   f"⏳ Yaklaşıyor: "
                   f"**{sum(1 for r in top if r['_kademe'] == 2)}** — "
                   f"'Mesafe' = pahalılık − 35 (negatif: bölgenin içinde).")
        cur = top[0]["_cur"]
        df = pd.DataFrame([
            {"Sıra": i + 1, "Durum": r["Durum"], "Ticker": r["Ticker"],
             "İsim": r["İsim"], "Fiyat": f"{cur}{r['Fiyat']:,.2f}",
             "Kalite": r["Kalite"], "Pahalılık": r["Pahalılık"],
             "Mesafe": r["Mesafe"],
             "52h": (f"%{r['52h']*100:.0f}" if r.get("52h") is not None
                     else "—"),
             "200G Trend": (f"%{r['Trend200']*100:+.0f}"
                            if r.get("Trend200") is not None else "—"),
             "Analist Pot.": (f"%{r['AnalistPot']*100:+.0f}"
                              if r.get("AnalistPot") is not None else "—"),
             "Beta": (f"{r['Beta']:.1f}" if r.get("Beta") is not None
                      else "—"),
             "PD": fmt_money(r["PD"], cur),
             "Öne Çıkan": r["Not"]}
            for i, r in enumerate(top)])
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇️ CSV indir",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name=f"yaklasanlar_{ts['pazar']}.csv",
                           mime="text/csv")
        # ══ B12-3 (KRİTİK) ═══════════════════════════════════════════
        # Bu liste ~500 gürültülü skorun EN İYİ 20'sidir — yani uç sıra
        # istatistikleridir ve seçim etkisiyle YUKARI YANLIDIR.
        # Harvey-Liu-Zhu (RFS 2016): yüzlerce faktör test edildiğinde
        # geleneksel t>2.0 çıtası çok gevşek, t>3.0 gerekiyor.
        # Novy-Marx (NBER 2015): ÇOK SİNYALLİ birleşimlerde yanlılık,
        # "n^k aday arasından en iyisini seçmeye" yakın büyüklükte.
        # Kaydırıcılar bunu AĞIRLAŞTIRIR: her ayar yeni bir denemedir ve
        # kullanıcı liste "hoşuna gidince" durur → belirtim arama.
        _n_ayar = st.session_state.get("_filtre_denemesi", 0)
        st.error(
            "📐 **İSTATİSTİKSEL DURUM — önce bunu oku.** Bu liste "
            "*test edilmiş bir strateji DEĞİL*, aday üretecidir. "
            "~500 skorun en iyi 20'si gösteriliyor; bu skorlar **seçim "
            "etkisiyle yukarı yanlıdır** (en iyi 20, tanım gereği "
            "gürültünün de en şanslı 20'sidir). Örneklem dışı hiçbir "
            "doğrulama YOKTUR."
            + (f" Bu oturumda **{_n_ayar} farklı filtre ayarı** denendin — "
               f"ne kadar çok ayar denenirse liste o kadar gürültüye "
               f"uydurulmuş olur (belirtim arama)."
               if _n_ayar > 3 else ""))
        st.caption("Bu bir İZLEME/aday listesidir, alım listesi değil — "
                   "yaklaşan her hisse derin analizden geçmeli; 'bölgede' "
                   "olmak alım emri değildir. Kısa ufukta (3-4 ay) '200G "
                   "Trend', 'Analist Pot.' ve 'Beta' sütunları BAĞLAMDIR: "
                   "trend altı ucuzlayan hisse daha da ucuzlayabilir, "
                   "yüksek beta geniş dalgalanma demektir — bunlar yön "
                   "sözü değildir. Yatırım tavsiyesi değildir.")
        sc1, sc2 = st.columns([2, 1])
        sec = sc1.selectbox("Derin analize gönderilecek aday",
                            [r["Ticker"] for r in top])
        if sc2.button("🔬 Derin Analize Gönder",
                      width="stretch"):
            st.session_state["_git_tkr"] = sec
            st.session_state["_git_mod"] = MOD_TEK
            st.rerun()
        st.caption("Gönderdikten sonra sol menüden **Analizi Çalıştır**'a "
                   "bas — tam motor (SEC çaprazı, Monte Carlo, kademeli "
                   "plan, Claude raporu) devreye girer.")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kapanis_seri(ticker: str, period: str = "1y"):
    """Portföy paneli için hafif, temettü/split düzeltmeli kapanış serisi."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker, session=_yf_session())
        # B3-1 NOTU: burada auto_adjust=True BİLEREK korunuyor. Bu seri
        # yalnız KORELASYON/getiri hesabında kullanılır; getiri ölçümünde
        # doğru baz TOPLAM GETİRİDİR. İşlem seviyeleri (destek/stop) için
        # kullanılan seri fetch_bundle'daki temettü-düzeltmesiz bazdır.
        h = _yf_retry(lambda: tk.history(period=period, auto_adjust=True),
                      tries=2, base_wait=1.0)
        if h is None or getattr(h, "empty", True) or "Close" not in h:
            return None
        c = h["Close"].dropna()
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        return c
    except Exception:
        return None


def portfoy_ozet(seriler, sektorler, risk_pct):
    """
    PORTFÖY RİSK ÖZETİ (bütünsel rapor G9) — saf, test edilebilir.
      1) 90 günlük getiri korelasyonu: yüksekse 'çeşitlendirdim' sanıp
         fiilen TEK pozisyon taşıyorsun demektir.
      2) Sektör yoğunlaşması (eşit ağırlık varsayımı [TÜRETİLMİŞ]).
      3) Toplam açık risk = pozisyon sayısı × işlem başına risk — her
         pozisyonda stop varsayımıyla; gap'te gerçek zarar stopu AŞABİLİR.
    Yön tahmini YOKTUR.
    """
    if not seriler or len(seriler) < 2:
        return None
    df = pd.DataFrame(seriler).dropna()
    if len(df) < 40:
        return None
    r = df.pct_change().dropna().iloc[-90:]
    if len(r) < 30:
        return None
    corr = r.corr()
    v = corr.values
    n = len(v)
    ort = float((v.sum() - n) / (n * (n - 1))) if n > 1 else None
    kolon = list(corr.columns)
    ciftler = []
    for i in range(n):
        for j in range(i + 1, n):
            ciftler.append((kolon[i], kolon[j], float(v[i][j])))
    ciftler.sort(key=lambda x: -x[2])
    sk = {}
    for t, s in (sektorler or {}).items():
        sk[s] = sk.get(s, 0) + 1
    toplam = sum(sk.values()) or 1
    sektor_pay = {s: c / toplam for s, c in sk.items()}
    max_sektor = (max(sektor_pay.items(), key=lambda x: x[1])
                  if sektor_pay else ("Bilinmiyor", 0.0))
    # ══ B12-4 (KRİTİK) ═══════════════════════════════════════════════
    # "toplam risk = N × işlem_riski" KORELASYONU TAMAMEN YOK SAYIYOR.
    # Doğru ifade portföy varyansından gelir; N eşit pozisyon, ortalama
    # ikili korelasyon ρ için:
    #     σp = σ × √( 1/N + (N−1)/N × ρ )
    # ÖLÇÜM (N=6, işlem başına %1):
    #     ρ=0.0 → gerçek %2.45 (kod 2.45 KAT abartıyor)
    #     ρ=0.3 → gerçek %3.87 (1.55 kat)
    #     ρ=0.5 → gerçek %4.58 (1.31 kat)
    #     ρ=0.7 → gerçek %5.20 (1.15 kat)
    #     ρ=1.0 → gerçek %6.00 (kod DOĞRU — ama bu ÇÖKÜŞ senaryosudur)
    # Yani N×risk bir BEKLENEN risk değil, TÜM STOPLAR AYNI ANDA
    # tetiklenirse geçerli olan ÜST SINIRDIR. İkisi ayrı gösterilir.
    _N = len(seriler)
    _rho = max(0.0, min(1.0, ort if ort is not None else 0.0))
    _carpan = (1.0 / _N + (_N - 1) / _N * _rho) ** 0.5
    _cokus = _N * float(risk_pct)
    _beklenen = _cokus * _carpan

    # B12-4b: 90 gözlemde korelasyonun hata payı
    _se = ((1 - _rho ** 2) / max(len(r) - 2, 1)) ** 0.5
    _ga = (max(-1.0, _rho - 1.96 * _se), min(1.0, _rho + 1.96 * _se))

    # B12-4c: sayı-tabanlı sektör payı yerine HHI + etkin sektör sayısı
    _hhi = sum(p * p for p in sektor_pay.values()) if sektor_pay else None
    _etkin = (1.0 / _hhi) if _hhi else None

    return {"gun": int(len(r)), "ort_korelasyon": round(_rho, 2),
            "korelasyon_ga95": (round(_ga[0], 2), round(_ga[1], 2)),
            "korelasyon_se": round(_se, 3),
            "en_yuksek_ciftler": ciftler[:5],
            "korelasyon_matrisi": corr.round(2),
            "sektor_pay": sektor_pay, "max_sektor": max_sektor,
            "sektor_hhi": round(_hhi, 3) if _hhi else None,
            "etkin_sektor_sayisi": round(_etkin, 1) if _etkin else None,
            "beklenen_risk_pct": round(_beklenen, 2),
            "cokus_ust_sinir_pct": round(_cokus, 2),
            "cesitlendirme_carpani": round(_carpan, 2),
            "toplam_acik_risk_pct": round(_cokus, 2),   # geriye uyumluluk
            "risk_notu": (
                f"İKİ AYRI SAYI: (1) BEKLENEN risk %{_beklenen:.2f} — "
                f"ρ={_rho:.2f} korelasyonla düzeltilmiş 1σ; (2) ÇÖKÜŞ ÜST "
                f"SINIRI %{_cokus:.2f} — tüm stoplar AYNI ANDA tetiklenirse. "
                f"Eski sürüm yalnız ikincisini gösterip 'toplam açık risk' "
                f"diyordu; bu, çeşitlendirilmiş bir portföyde riski "
                f"{1/_carpan:.2f} kat ABARTIR."),
            "korelasyon_notu": (
                f"{len(r)} gözlemde ρ={_rho:.2f}'nin %95 güven aralığı "
                f"[{_ga[0]:.2f}, {_ga[1]:.2f}] — yani 0.70 eşiği ile 0.60 "
                f"İSTATİSTİKSEL OLARAK AYIRT EDİLEMEZ. Ayrıca korelasyon "
                f"KRİZDE YÜKSELİR: 2008'de ortalama ikili korelasyon "
                f"~%40'tan ~%70'e çıktı; sektör ETF'lerinde 2008'de 0.83, "
                f"Mart 2020'de 0.92 görüldü. Bugünkü ılımlı bir korelasyon, "
                f"kriz davranışının GÜVENCESİ DEĞİLDİR — çeşitlendirme tam "
                f"da en çok ihtiyaç duyulduğu anda kaybolur."),
            "varsayim_notu": (
                "EŞİT AĞIRLIK ve EŞİT RİSK varsayılmıştır. Gerçekte eşit "
                "dolar ağırlık eşit RİSK katkısı vermez — yüksek oynaklıklı "
                "isimler varyansa hâkim olur. Pozisyon boyutların ya da "
                "oynaklıklar belirgin farklıysa gerçek yoğunlaşma daha "
                "yüksektir.")}



# ═══════════ 💰 TAHSİS — ters-vol + sınırlı tilt (araştırma uyg.) ═══════════
_TH_IBARE = ("Bu bölüm, girdiğin sayılara dayalı EĞİTİM AMAÇLI hesaplanmış "
             "bir çerçevedir; kişiye özel yatırım tavsiyesi değildir. "
             "Ağırlık/tutar/adet, seçilen varsayımlarla mekanik üretilir; "
             "getiri-kayıp taahhüdü yoktur. Karar ve sorumluluk sana aittir.")


def _th_agirliklar(sig, skor, kal, k, tavan, minw, yontem="tersvol",
                   pah=None, sec=None):
    # taban: ters-vol; tilt: clip(1+k·z,0.5,1.5); DÜZELTME: N-duyarlı
    # tavan (max(tavan,1/N)) + kilitle-ve-dağıt (kırpılan SABİTLENİR).
    syms = list(sig)
    _poz = [v for v in sig.values() if v is not None and v > 1e-6]
    med = float(np.median(_poz)) if _poz else 0.02   # A-1 testi: NaN/negatif kökü
    inv = {s: 1.0 / (sig[s] if sig[s] and sig[s] > 1e-6 else med)
           for s in syms}
    if yontem == "bilesik":           # çok-ölçütlü seçim puanı ÷ oynaklık
        pah = pah or {}
        sec = sec or {}
        g = {}
        for s in syms:
            q = float(kal[s]) if kal.get(s) is not None else 50.0
            u = 100.0 - (float(pah[s]) if pah.get(s) is not None else 50.0)
            b = (float(sec[s]) if sec.get(s) is not None
                 else 0.6 * q + 0.4 * u)
            if kal.get(s) is not None and kal[s] < 40:
                b *= 0.5                  # düşük kalite: telafisiz yarım
            g[s] = max(b, 5.0) * inv[s]
    elif yontem == "kalite":          # saf kalite (risk görmez — uyarılı)
        g = {s: max(float(kal[s]) if kal.get(s) is not None else 50.0, 5.0)
             for s in syms}
    elif yontem == "hibrit":          # kalite / oynaklık (risk-farkında)
        g = {s: (max(float(kal[s]) if kal.get(s) is not None else 50.0, 5.0)
                 * inv[s]) for s in syms}
    else:                              # ters-volatilite (varsayılan)
        g = dict(inv)
    tp = sum(g.values())
    w0 = {s: g[s] / tp for s in syms}
    w = dict(w0)
    gecerli = [v for v in skor.values() if v is not None]
    mu = float(np.mean(gecerli)) if gecerli else 0.0
    sd = float(np.std(gecerli)) if len(gecerli) > 1 else 0.0
    tilt = {}
    for s in syms:
        z = ((skor[s]-mu)/sd) if (skor[s] is not None and sd > 1e-9) else 0.0
        tilt[s] = float(np.clip(1.0 + k*z, 0.5, 1.5))
        w[s] *= tilt[s]
    n = sum(w.values()) or 1.0
    w = {s: v/n for s, v in w.items()}
    notlar = {s: [] for s in syms}

    def _kirp(w):
        akt = [s for s in w if w[s] > 0]
        tv = max(tavan, 1.0/max(len(akt), 1) + 1e-9)   # uygulanabilir tavan
        sabit = set()
        for _ in range(24):
            fazla = [s for s in akt if s not in sabit and w[s] > tv + 1e-9]
            if not fazla:
                break
            for s in fazla:
                w[s] = tv
                sabit.add(s)
                if "tavan" not in notlar[s]:
                    notlar[s].append("tavan")
            serbest = 1.0 - tv*len(sabit)
            serb = [s for s in akt if s not in sabit]
            kt = sum(w[s] for s in serb)
            if serbest <= 1e-9 or kt <= 1e-12 or not serb:
                for s in akt:
                    w[s] = 1.0/len(akt)
                return w, tv
            for s in serb:
                w[s] = w[s]/kt*serbest
        return w, tv

    w, tv = _kirp(w)
    for s in list(w):                       # anlamsız küçükleri ele
        if 0 < w[s] < minw:
            notlar[s].append("min-altı→0")
            w[s] = 0.0
    n = sum(w.values()) or 1.0
    w = {s: v/n for s, v in w.items()}
    w, tv = _kirp(w)                        # eleme sonrası tavanı yeniden
    if tv > tavan + 1e-6:
        for s in w:
            if w[s] > 0 and "tavan(N)" not in notlar[s] and "tavan" in notlar[s]:
                notlar[s][notlar[s].index("tavan")] = f"tavan(1/N=%{tv*100:.0f})"
    return w, tilt, notlar, w0


def render_tahsis(pf, oz):
    st.markdown("---")
    st.markdown("## 💰 Tahsis — Hesaplanan Çerçeve")
    st.caption("Yöntem: ters-volatilite tabanı (oynağa az, sakine çok) + "
               "kalite−pahalılık skorundan ×0.5–1.5 bandına KIRPILMIŞ eğim "
               "+ pozisyon tavanı + nakit tamponu. Ortalama-varyans/Kelly "
               "bilerek YOK: az hissede tahmin hatası onları 1/N'in bile "
               "gerisine düşürür (DeMiguel 2009; Michaud 1989). Taban "
               "yöntemini aşağıdan sen seçebilirsin.")
    c1, c2, c3 = st.columns(3)
    sermaye = c1.number_input("Toplam sermaye ($)", 500.0, 1e8, 10000.0,
                              500.0, key="th_ser")
    istah = c2.radio("Risk iştahı", ["Temkinli", "Dengeli", "Agresif"],
                     index=1, horizontal=True, key="th_ist")
    stop_pct = c3.slider("Stop mesafesi (tümüne, %)", 5.0, 25.0, 12.0, 0.5,
                         key="th_stp",
                         help="[YAKLAŞIK] Tek-hisse ekranındaki ATR-stop "
                              "daha isabetlidir; burada kaba risk katmanı "
                              "için tek oran uygulanır.")
    yontem_ad = st.radio("Ağırlık tabanı",
                         ["Bileşik çok-ölçütlü 🎯 (ciddi dağılım)",
                          "Ters-vol + kalite eğimi",
                          "Kalite/Oynaklık (hibrit)",
                          "Saf kalite ⚠️"],
                         horizontal=True, key="th_yon")
    yontem = ("kalite" if yontem_ad.startswith("Saf")
              else "hibrit" if yontem_ad.startswith("Kalite/")
              else "bilesik" if yontem_ad.startswith("Bileşik")
              else "tersvol")
    if yontem == "bilesik":
        st.caption("🎯 Bileşik: tabanda ÇOK-ölçütlü SEÇİM puanı (kalite "
                   "bileşenleri + ucuzluk + telafi-etmeyen vetolar; seçim "
                   "puanı yoksa 0.6·kalite+0.4·ucuzluk) ÷ oynaklık. Düşük "
                   "kalite (<40) YARI ağırlığa düşer — hiçbir tek sayı tek "
                   "başına belirlemez; döküm aşağıdaki 'Neden bu "
                   "ağırlık?'ta hisse hisse açıktır.")
    if yontem == "kalite":
        st.warning("Saf kalite tabanı OYNAKLIĞI görmez: kaliteli ama çok "
                   "oynak bir hisse portföy riskini domine edebilir — "
                   "σp metriğini mutlaka izle. Pahalılık da tabana girmez.")
    elif yontem == "hibrit":
        st.caption("Hibrit: ağırlık ∝ kalite ÷ oynaklık — kaliteyi sever, "
                   "riski de dengeler. Pahalılık tabana girmez; eğim "
                   "kapalıdır (çifte sayım olmasın).")
    k, tavan, minw, tampon = {"Temkinli": (0.0, 0.10, 0.05, 0.10),
                              "Dengeli": (0.6, 0.20, 0.04, 0.05),
                              "Agresif": (1.0, 0.25, 0.03, 0.02)}[istah]
    # Az hissede tavan, seçilen yöntemi tamamen ezebilir (N×tavan≤%100 →
    # zorunlu eşitlik). Kontrol kullanıcıda: kaydırıcı + kaliteye %40 öneri.
    _tv_oner = 40 if yontem == "kalite" else int(tavan * 100)
    if st.session_state.get("_th_yon_onceki") != yontem:
        st.session_state["_th_yon_onceki"] = yontem
        st.session_state["th_tav"] = _tv_oner   # geçiş anında öneriyi uygula
    tavan = st.slider("Tek-pozisyon tavanı (%)", 10, 60, _tv_oner, 5,
                      key="th_tav",
                      help="Hisse sayısı × tavan %100'ü geçmiyorsa "
                           "ağırlıklar EŞİTLİĞE zorlanır ve taban yöntemi "
                           "görünmez olur. Kaliteye göre ayrışma istiyorsan "
                           "tavanı yükselt (yoğunlaşma riski senin "
                           "bilinçli tercihin olur).") / 100.0
    if not st.checkbox("Bu çıktının kişiye özel tavsiye DEĞİL, hesaplanmış "
                       "bir eğitim çerçevesi olduğunu ve kararların bana "
                       "ait olduğunu anladım.", key="th_ok"):
        st.info("Tahsis tablosu için yukarıdaki onayı işaretle.")
        st.caption(_TH_IBARE)
        return
    seriler = pf["seriler"]
    syms = sorted(seriler)
    sig, fiy, skor = {}, {}, {}
    kal_d, pah_d, sec_d = {}, {}, {}
    for s in syms:
        ser = seriler[s].dropna()
        r = ser.pct_change().dropna().tail(90)
        sig[s] = float(r.std()) if len(r) >= 40 else None
        fiy[s] = float(ser.iloc[-1])
        try:
            sk = hizli_skorlar(hizli_veri(s))
            kal_d[s] = sk["kalite"]
            pah_d[s] = sk["pahalilik"]
            sec_d[s] = sk.get("secim")
            skor[s] = ((sk["kalite"] if sk["kalite"] is not None else 50)
                       - (sk["pahalilik"] if sk["pahalilik"] is not None
                          else 50))
        except Exception:
            kal_d[s], pah_d[s], sec_d[s], skor[s] = None, None, None, None
    k_eff = k if yontem == "tersvol" else 0.0
    w, tilt, notlar, w0 = _th_agirliklar(sig, skor, kal_d, k_eff,
                                         tavan, minw, yontem,
                                         pah=pah_d, sec=sec_d)
    _akt = [s for s in w if w[s] > 0]
    if (len(_akt) > 1
            and max(w[s] for s in _akt) - min(w[s] for s in _akt) < 1e-6
            and abs(max(w[s] for s in _akt) - 1.0/len(_akt)) < 1e-6):
        st.info(f"ℹ️ {len(_akt)} hisse × %{tavan*100:.0f} tavan, ağırlıkları "
                f"matematiksel olarak EŞİTLİĞE zorladı — seçtiğin taban "
                f"yönteminin ayrışması için yukarıdan tavanı yükselt ya da "
                f"hisse sayısını artır.")
    yat = sermaye * (1.0 - tampon)
    sat, top_t, top_r = [], 0.0, 0.0
    for s in syms:
        tut = w[s] * yat
        ad = int(tut // fiy[s]) if fiy[s] > 0 else 0
        g = ad * fiy[s]
        rk = g * stop_pct / 100.0
        top_t += g
        top_r += rk
        nt = list(notlar[s])
        if sig[s] is None:
            nt.append("σ<40g→medyan varsayıldı")
        if skor[s] is None:
            nt.append("skor yok→eğimsiz")
        if kal_d[s] is not None and kal_d[s] < 40:
            nt.append("⚪düşük kalite")
        sat.append({"Hisse": s, "Ağırlık %": round(w[s] * 100, 1),
                    "Tutar $": round(g, 0), "Adet": ad,
                    "Stop'ta kayıp $": round(rk, 0),
                    "5/25 bandı ±pp": round(min(5.0, w[s] * 25), 1),
                    "Not": ", ".join(nt) or "—"})
    df = pd.DataFrame(sat).sort_values("Ağırlık %", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    nakit = sermaye - top_t
    # σp: tam kovaryans w'Σw (90g korelasyon matrisi + σ vektörü)
    try:
        R = oz["korelasyon_matrisi"].loc[syms, syms].astype(float).values
        sv = np.array([sig[s] if sig[s] else np.nanmedian(
            [x for x in sig.values() if x]) for s in syms])
        wv = np.array([w[s] for s in syms])
        sp = float(np.sqrt(wv @ (np.outer(sv, sv) * R) @ wv))
        sp_y = sp * np.sqrt(252) * 100
    except Exception:
        sp_y = None
    neff = 1.0 / sum(v * v for v in w.values() if v) if any(w.values()) else 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Yatırılan", f"${top_t:,.0f}")
    m2.metric("Nakit tamponu", f"${nakit:,.0f}",
              f"%{nakit/sermaye*100:.1f}", delta_color="off")
    m3.metric("Hepsi stop olursa", f"−${top_r:,.0f}",
              f"%{top_r/sermaye*100:.1f} sermaye", delta_color="off")
    m4.metric("σp (yıllık, ρ'lu) [YAKLAŞIK]",
              f"%{sp_y:.0f}" if sp_y else "—",
              f"N_eff≈{neff:.1f}", delta_color="off")
    if len([s for s in w if w[s] > 0]) < 4:
        st.warning("Aktif pozisyon < 4 — çeşitlendirme sınırlı; tek şirket "
                   "haberi sepeti sürükler.")
    if any(kal_d.get(s) is not None and kal_d[s] < 40 and w[s] > 0
           for s in syms):
        st.warning("⚪ Düşük-kalite kademesinde hisse(ler) var: ucuzluk "
                   "kaliteyi TELAFİ ETMEZ; ağırlıkları bilinçli gözden "
                   "geçir.")
    with st.expander("🔍 Neden bu ağırlık? (taban → eğim → kırpma)"):
        st.dataframe(pd.DataFrame(
            [{"Hisse": s,
              "σ90 %g": round((sig[s] or 0) * 100, 2),
              "Kalite": kal_d.get(s), "Pahalılık": pah_d.get(s),
              "Seçim": sec_d.get(s),
              "Taban %": round(w0[s] * 100, 1),

              "Eğim ×": round(tilt[s], 2),
              "Son %": round(w[s] * 100, 1)} for s in syms]),
            hide_index=True, width="stretch")
        st.caption("Eğim = kalite−pahalılık z-skoru; ×0.5–1.5 dışına "
                   "ÇIKAMAZ (Michaud: agresif eğim tahmin hatasını "
                   "büyütür). Tavan/min sonrası yeniden normalize edilir.")
    st.download_button("⬇️ Tahsis (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       "tahsis.csv", "text/csv", key="th_csv")
    st.caption("Yeniden dengeleme: 5/25 kuralı — ağırlık, hedefinden "
               "mutlak 5 puan YA DA göreli %25 (küçük olan) saparsa "
               "dengele (Daryanani 2008). " + _TH_IBARE)
# ═══════════════════════ /TAHSİS ═══════════════════════


def render_portfoy():
    st.markdown("# 🧺 Portföy Riski")
    st.caption("Rapor bulgusu (G9): tek-hisse analizi ne kadar iyi olursa "
               "olsun risk POZİSYONLARIN TOPLAMINDA doğar. Bu panel yön "
               "tahmini yapmaz; üç şeyi ölçer: pozisyonlar birlikte mi "
               "hareket ediyor (90g korelasyon), sektör yoğunlaşması ve "
               "toplam açık risk bütçesi.")
    ham = st.text_input("Elindeki/planladığın pozisyonlar (2-10, "
                        "virgül/boşluk)", value="", key="pf_giris",
                        help="Ör: AAPL, MSFT, JPM")
    tickers = tuple(dict.fromkeys(
        t.strip().upper() for t in ham.replace(",", " ").split()
        if t.strip()))[:10]
    risk_pct = st.slider("İşlem başına risk (%)", 0.25, 3.0, 1.0, 0.25,
                         key="pf_risk",
                         help="Stop planındaki risk kuralıyla aynı mantık; "
                              "toplam açık risk = pozisyon sayısı × bu "
                              "oran (eşit risk varsayımı).")
    if st.button("🧺 Hesapla", type="primary", width="stretch",
                 disabled=len(tickers) < 2):
        seriler, sektorler, yok = {}, {}, []
        prog = st.progress(0.0, text="Veri çekiliyor…")
        for i, t in enumerate(tickers):
            c = fetch_kapanis_seri(t)
            d = hizli_veri(t)
            prog.progress((i + 1) / len(tickers), text=t)
            if c is None or len(c) < 60:
                yok.append(t)
                continue
            seriler[t] = c
            sektorler[t] = (d or {}).get("sector") or "Bilinmiyor"
        prog.empty()
        st.session_state["portfoy"] = {"seriler": seriler,
                                       "sektorler": sektorler,
                                       "yok": yok,
                                       "risk_pct": float(risk_pct)}
    pf = st.session_state.get("portfoy")
    if not pf:
        return
    if pf.get("yok"):
        st.caption("Veri alınamayan semboller (atlandı): "
                   + ", ".join(pf["yok"]))
    oz = portfoy_ozet(pf["seriler"], pf["sektorler"], pf["risk_pct"])
    if not oz:
        st.warning("Korelasyon için en az 2 sembolden ~2 aylık ortak veri "
                   "gerekli.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Ortalama Korelasyon (90g)",
              f"{oz['ort_korelasyon']:.2f}",
              f"GA95 {oz['korelasyon_ga95'][0]:.2f}–{oz['korelasyon_ga95'][1]:.2f}",
              delta_color="off",
              help="0'a yakın = gerçek çeşitlendirme; 0.7+ = fiilen tek pozisyon. GA95: 90 gözlemin hata payı.")
    c2.metric("En Yoğun Sektör", f"{oz['max_sektor'][0]}",
              f"%{oz['max_sektor'][1]*100:.0f}", delta_color="off")
    # O2: motorun B12-4 çıktıları ekranda — İKİ AYRI SAYI.
    c3.metric("Beklenen Risk (ρ-düzeltmeli 1σ)",
              f"%{oz['beklenen_risk_pct']:.1f}",
              f"çöküş üst sınırı %{oz['cokus_ust_sinir_pct']:.1f}",
              delta_color="off",
              help="Beklenen: σp = σ·√(1/N+(N−1)ρ/N). Üst sınır: TÜM "
                   "stoplar aynı anda tetiklenirse (N×risk); gap'te bu "
                   "bile aşılabilir.")
    st.caption(f"Çeşitlendirme çarpanı ×{oz['cesitlendirme_carpani']:.2f}"
               f" · Sektör HHI {oz.get('sektor_hhi')}"
               f" · Etkin sektör ≈{oz.get('etkin_sektor_sayisi')}")
    st.dataframe(oz["korelasyon_matrisi"], width="stretch")
    if oz["ort_korelasyon"] is not None and oz["ort_korelasyon"] >= 0.70:
        st.error("⚠️ Ortalama korelasyon 0.70+ — bu sepet fiilen TEK "
                 "pozisyon gibi davranır: aynı gün hepsi düşebilir, "
                 "stoplar aynı anda tetiklenebilir. Çeşitlendirme "
                 "yanılsamasına dikkat.")
    for t1, t2, c_ in oz["en_yuksek_ciftler"]:
        if c_ >= 0.85:
            st.warning(f"{t1} ↔ {t2} korelasyon {c_:.2f} — ikisini birden "
                       f"taşımak riski neredeyse ikiye katlar, "
                       f"çeşitlendirmez.")
    if oz["max_sektor"][1] > 0.5:
        st.warning(f"Sektör yoğunlaşması: pozisyonların "
                   f"%{oz['max_sektor'][1]*100:.0f}'i "
                   f"{oz['max_sektor'][0]} — tek sektör şoku sepetin "
                   f"tamamını vurur.")
    if oz["cokus_ust_sinir_pct"] > 6:
        st.warning(f"ÇÖKÜŞ ÜST SINIRI %{oz['cokus_ust_sinir_pct']:.1f} "
                   f"(beklenen %{oz['beklenen_risk_pct']:.1f}) — %6 üstü "
                   f"bütçe eşzamanlı stoplarda sermayeyi hızlı eritir; "
                   f"pozisyon sayısını ya da işlem riskini azalt.")
    with st.expander("📐 Bu iki sayı nasıl hesaplandı? (oku — önemli)"):
        st.caption(oz["risk_notu"])
        st.caption(oz["korelasyon_notu"])
        st.caption(oz["varsayim_notu"])
    st.caption("Sektörler: " + " · ".join(
        f"{s} %{p*100:.0f}"
        for s, p in sorted(oz["sektor_pay"].items(), key=lambda x: -x[1])))
    st.caption("Eşit ağırlık/eşit risk varsayımı [TÜRETİLMİŞ]; korelasyon "
               "geçmişe bakar ve krizde YÜKSELİR. Yatırım tavsiyesi "
               "değildir.")
    render_tahsis(pf, oz)



# ── KIYAS: veri toplama + arayüz ─────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def kiyas_revizyon_orani(t: str):
    """Analist revizyon genişliği — Chan-Jegadeesh-Lakonishok sinyali.

    Ham net sayı DEĞİL ORAN: (yukarı−aşağı)/(yukarı+aşağı). Ham sayı
    analist kapsamasına bağımlıdır (B6-7 bulgusu).
    """
    try:
        import yfinance as yf
        er = yf.Ticker(t, session=_yf_session()).eps_revisions
        if er is None or getattr(er, "empty", True):
            return None
        c = {str(x).lower(): x for x in er.columns}
        u, d = c.get("uplast30days"), c.get("downlast30days")
        if not u or not d:
            return None
        yu = float(pd.to_numeric(er[u], errors="coerce").fillna(0).sum())
        asa = float(pd.to_numeric(er[d], errors="coerce").fillna(0).sum())
        return ((yu - asa) / (yu + asa)) if (yu + asa) > 0 else None
    except Exception:
        return None


def kiyas_veri_topla(t, d, kapanis):
    """Bir aday için kıyas motorunun beklediği alanları hazırla."""
    g = lambda k: _num((d or {}).get(k))
    h = {}
    # 12-1 momentum: son ayı ATLA (kısa vade dönüşü ayıklamak için)
    try:
        c = kapanis.dropna()
        if len(c) >= 253:
            h["momentum_12_1"] = float(c.iloc[-22]) / float(c.iloc[-253]) - 1
    except Exception:
        pass
    # 52 hafta zirveye yakınlık
    hi = g("fiftyTwoWeekHigh")
    fi = g("fiyat")
    if hi and fi and hi > 0:
        h["zirve_yakinlik"] = fi / hi
    h["acik_satis"] = g("shortPercentOfFloat")
    h["revizyon_orani"] = kiyas_revizyon_orani(t)
    # veto girdileri
    h["fcf"] = g("freeCashflow")
    # ── ZİNCİR ONARIMI (denetim Bulgu 2): '_hacim' anahtarını HİÇBİR
    # üretici yazmıyordu → likidite vetosu ölüydü. yfinance hacim
    # alanları ADET döndürür (averageVolume / averageDailyVolume3Month /
    # averageDailyVolume10Day — resmi doküman); dolar hacmi = adet×fiyat.
    try:
        _adet = (g("averageVolume") or g("averageDailyVolume3Month")
                 or g("averageDailyVolume10Day"))
        if _adet and fi:
            h["gunluk_hacim_usd"] = float(_adet) * float(fi)
    except Exception:
        pass
    # Kazanç-takvimi vetosu da beslenmiyordu: info'daki earningsTimestamp
    # (epoch sn) varsa gün sayısına çevrilir.
    try:
        _ets = ((d or {}).get("earningsTimestamp")
                or (d or {}).get("earningsTimestampStart"))
        if _ets:
            _gk = int((pd.Timestamp(int(_ets), unit="s")
                       - pd.Timestamp.now()).days)
            if -1 <= _gk <= 180:
                h["kazanc_gun_kaldi"] = _gk
    except Exception:
        pass
    h["_sektor"] = (d or {}).get("sector")
    h["_isim"] = (d or {}).get("isim", t)
    h["_fiyat"] = fi
    return h


def render_kiyas_v2(veriler, kapanislar):
    """Kanıta dayalı başa baş karşılaştırma arayüzü."""
    hisseler = {v["t"]: kiyas_veri_topla(v["t"], v.get("_ham"),
                                         kapanislar.get(v["t"]))
                for v in veriler if kapanislar.get(v["t"]) is not None}
    if len(hisseler) < 2:
        st.warning("Karşılaştırma için en az 2 hissede fiyat geçmişi "
                   "gerekiyor.")
        return
    r = kiyas_motoru(hisseler)
    if not r:
        return

    # ── Denetim Bulgu 5: _isim/_sektor toplanıyor ama gösterilmiyordu —
    # kullanıcı ticker'ın hangi şirkete/sektöre denk geldiğini kıyas
    # ekranında doğrulayamıyordu. Ayrıca tüm adaylar aynı sektördense
    # sıralama sektör-içi görelidir; bunu söylemek dürüstlük gereği.
    _adlar = " · ".join(
        f"{t}: {(hisseler[t] or {}).get('_isim') or t}"
        + (f" ({hisseler[t]['_sektor']})"
           if (hisseler[t] or {}).get("_sektor") else "")
        + (f" — ${hisseler[t]['_fiyat']:,.2f}"
           if (hisseler[t] or {}).get("_fiyat") else "")
        for t in hisseler)
    if _adlar:
        st.caption("Adaylar — " + _adlar)
    _sekt = {(hisseler[t] or {}).get("_sektor") for t in hisseler
             if (hisseler[t] or {}).get("_sektor")}
    if len(hisseler) >= 2 and len(_sekt) == 1:
        st.caption("⚠️ Adayların hepsi AYNI sektörden — kıyas sektör-içi "
                   "göreli sıralamadır; sektörün kendisi yanlışsa hepsi "
                   "birlikte düşer.")

    # ── 0) DÜRÜSTLÜK ÖNCE ────────────────────────────────────────────
    st.error(f"⚖️ **ÖNCE BUNU OKU.** {r['durustluk']}")

    # ── 1) VETO ─────────────────────────────────────────────────────
    vet = {t: v for t, v in r["vetolar"].items()
           if v["sert"] or v["yumusak"] or v["takvim"]}
    if vet:
        st.markdown("#### 🚫 Veto ve Uyarılar (sıralamadan ÖNCE)")
        for t, v in vet.items():
            for s in v["sert"]:
                st.error(f"**{t} — DİSKALİFİYE:** {s}")
            for s in v["yumusak"]:
                st.warning(f"**{t}:** {s}")
            for s in v["takvim"]:
                st.info(f"**{t}:** {s}")

    # ── Denetim Bulgu 2: bu hafif-veri yolunda hangi vetolar ÇALIŞAMADI?
    # (Boş veto listesi "temiz" değil "denetlenemedi" demek olabilir.)
    _kap = sorted({s for v in r["vetolar"].values()
                   for s in (v.get("kapsanmayan") or [])})
    if _kap:
        st.caption("🔎 Bu hafif-veri yolunda DENETLENEMEYEN vetolar: "
                   + " · ".join(_kap) + ". Kesin karar öncesi adayı Tek "
                   "Hisse Analizi'nden geçir — restatement/going-concern/"
                   "Altman/Beneish ancak orada görünür.")

    if len(r.get("yarisan") or []) < 2:
        st.warning(r.get("hukum", "Karşılaştırılacak aday kalmadı."))
        return

    # ── 2) KADEMELER (kesin sıra DEĞİL) ─────────────────────────────
    st.markdown("#### 🏁 Kademeler")
    st.caption("Kesin 1-2-3 yerine KADEME: aynı kademedeki isimler bu "
               "ufukta istatistiksel olarak ayırt edilemez.")
    for i, kd in enumerate(r["kademeler"], 1):
        isim = " · ".join(f"**{t}**" for t in kd)
        if len(kd) > 1:
            st.info(f"**{i}. kademe** — {isim}  \n"
                    f"↳ *bu isimler ayırt edilemiyor; aralarındaki fark "
                    f"gürültü sınırının altında*")
        else:
            st.success(f"**{i}. kademe** — {isim}")

    if not r["siralama_saglam"]:
        st.warning("⚠️ İki bağımsız sıralama kuralı (Borda ve Copeland) "
                   "FARKLI sıra veriyor — sıralama kırılgan, kademelere "
                   "güven, sıraya güvenme.")

    # ── 3) P(#1) — düzse bu bir BULGUDUR ────────────────────────────
    st.markdown("#### 🎲 'Birinci olma' olasılığı")
    pb = pd.DataFrame([
        {"Hisse": t,
         "P(#1)": (r["bootstrap"][t]["p_birinci"] or 0),
         "Ort. sıra": r["bootstrap"][t]["ortalama_sira"],
         "Sıra %95 GA": f"{r['bootstrap'][t]['sira_ga95'][0]}–"
                        f"{r['bootstrap'][t]['sira_ga95'][1]}",
         "Bileşik z": round(r["bilesik"][t], 2)}
        for t in r["yarisan"]])
    st.dataframe(pb.sort_values("P(#1)", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"P(#1)": st.column_config.ProgressColumn(
                     "P(#1)", min_value=0.0, max_value=1.0,
                     format="%.0f%%")})
    st.warning(
        "⚠️ **Bu sütunun NE OLDUĞUNU karıştırma.** P(#1), *sinyalleri "
        "doğru ölçtüysem sıralama bu* demektir — *bu hisse gerçekten daha "
        "çok kazandırır* DEMEK DEĞİLDİR. Sinyal ile gelecek getiri "
        "arasındaki bağ zayıf olduğu için gerçek getiri üstünlüğü "
        "olasılığı çok daha düşüktür; aşağıdaki ikili tabloya bak.")

    # ── ASIL SORULAN SORU: gerçekten daha çok kazandırır mı? ─────────
    st.markdown("#### 💰 \"Hangisi daha çok kazandırır?\" — kalibre cevap")
    sat_g = []
    for ad, v in r["ikili"].items():
        if v.get("p_getiri_ustunlugu") is None:
            continue
        p = v["p_getiri_ustunlugu"]
        sat_g.append({
            "Karşılaşma": ad,
            "Sinyal önde": v["onde"] or "—",
            "Sinyal farkı (SD)": f"{abs(v['z_farki']):.2f}",
            "P(önde olan daha çok kazanır)": f"%{p*100:.1f}",
            "Yorum": ("neredeyse yazı-tura" if p < 0.52 else
                      "hafif eğilim" if p < 0.55 else "belirgin eğilim")})
    if sat_g:
        st.dataframe(pd.DataFrame(sat_g), hide_index=True, width="stretch")
        st.caption(
            "Bu olasılıklar, sinyal farkının GERÇEK GETİRİYE çevrilmesiyle "
            "hesaplandı: P = Φ(ρ·Δz/√(2(1−ρ²))), ρ = harmanlanmış sinyalin "
            "bilgi katsayısı (~0.06). Neden bu kadar 50'ye yakın: 3-4 aylık "
            "ufukta bir hissenin hareketinin %65-85'i firmaya özgüdür ve "
            "hiçbir sinyalle öngörülemez. **Büyük bir sinyal farkı bile "
            "getiri üstünlüğünü ancak ~%55'e taşır.** Bu, aracın "
            "eksikliği değil piyasanın gerçeğidir.")
    _mx = max((r["bootstrap"][t]["p_birinci"] or 0) for t in r["yarisan"])
    if _mx < 0.55:
        st.caption("📊 Çubuklar DÜZ — bu bir kusur değil BULGUDUR: "
                   "veriler hiçbir adayı belirgin öne çıkarmıyor.")

    # ── 4) NEDEN — sinyal sinyal ────────────────────────────────────
    st.markdown("#### 🔍 Her isim neden orada?")
    sat = []
    for t in r["yarisan"]:
        d = {"Hisse": t}
        for s in SINYAL_AGIRLIK:
            z = r["z"][t].get(s)
            d[SINYAL_ADI[s]] = (f"{z:+.2f}" if z is not None else "—")
        d["kapsam"] = f"{r['kapsam'][t]}/{len(SINYAL_AGIRLIK)}"
        sat.append(d)
    st.dataframe(pd.DataFrame(sat), hide_index=True, width="stretch")
    st.caption("Değerler SEKTÖR-NÖTR z-skorudur (geniş evrene göre, "
               "aday kümesine göre DEĞİL — 3 hissede aday-içi yüzdelik "
               "zorunlu olarak 0/50/100 çıkar ve sahte ayrışma üretir). "
               f"Referans: {r['referans_kaynak']}.")

    # ── 5) İKİLİ ────────────────────────────────────────────────────
    with st.expander("⚔️ İkili karşılaştırmalar (sinyal sinyal)"):
        for ad, v in r["ikili"].items():
            k_ = v["sayac"]
            st.markdown(f"**{ad}** — {k_['a']} / {k_['b']} "
                        f"({k_['berabere']} berabere)")
            st.dataframe(pd.DataFrame([
                {"Sinyal": x["ad"], "Kazanan": x["sonuc"],
                 "Fark (SD)": ("—" if x["fark"] is None
                               else f"{x['fark']:+.2f}")}
                for x in v["detay"]]), hide_index=True, width="stretch")
        st.caption(f"Fark {BERABERLIK_SD} standart sapmanın altındaysa "
                   f"BERABERE sayılır — bu ufukta daha küçük bir fark "
                   f"gürültüden ayırt edilemez.")

    # ── SENARYO: hangi koşulda hangisi ──────────────────────────────
    # senaryo_farki() yazılmıştı ama HİÇ ÇAĞRILMIYORDU → kullanıcının
    # istediği "hangisi hangi koşulda kazanır" çıktısı hiç üretilmiyordu.
    _get = {}
    for _t in r["yarisan"]:
        _s = kapanislar.get(_t)
        if _s is not None and len(_s) > 120:
            _get[_t] = _s.pct_change().dropna().values[-260:]
    if len(_get) >= 2:
        _pk = fetch_endeks("^GSPC", "2y")
        _fk = {}
        if _pk is not None and len(_pk) > 120:
            _fk["piyasa"] = _pk.pct_change().dropna().values[-260:]
        if _fk:
            _n = min([len(v) for v in list(_get.values()) + list(_fk.values())])
            _sen = senaryo_farki({k_: v[-_n:] for k_, v in _get.items()},
                                 {k_: v[-_n:] for k_, v in _fk.items()})
            if _sen and not _sen.get("yetersiz"):
                st.markdown("#### 🌦️ Hangisi hangi koşulda?")
                st.dataframe(pd.DataFrame([
                    {"Karşılaşma": _ad, **{f"{_f} farkı": _v
                                           for _f, _v in _d.items()}}
                    for _ad, _d in _sen["farklar"].items()]),
                    hide_index=True, width="stretch")
                st.caption("Pozitif fark: ilk isim o faktöre DAHA duyarlı. "
                           + _sen["not"])
            elif _sen:
                st.caption("🌦️ Senaryo: " + _sen.get("not", ""))

    # ── 6) BOYUTLANDIRMA ────────────────────────────────────────────
    st.markdown("#### ⚖️ Tek kazanan yerine: kanaate göre dağıtım")
    st.caption("İkili isabet ~%52 iken tüm parayı 'birinci'ye koymak, "
               "kenarın taşıyabileceğinden fazla iddiadır. Küçük kenarı "
               "yaymak daha savunulabilir.")
    st.dataframe(pd.DataFrame([
        {"Hisse": t, "Öneri payı": f"%{v*100:.0f}",
         "Durum": ("VETO" if (r["vetolar"][t]["diskalifiye"]) else "—")}
        for t, v in sorted(r["boyutlandirma"].items(),
                           key=lambda x: -x[1])]),
        hide_index=True, width="stretch")

    st.info("ℹ️ " + r["kapsam_disi"])
    st.caption("Bu bir disiplin çerçevesidir, yatırım tavsiyesi değildir.")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_endeks(ticker: str, period: str = "10y"):
    """Endeks kapanış serisi (^IXIC, ^NDX, ^GSPC, ^VIX)."""
    try:
        import yfinance as yf
        h = _yf_retry(lambda: yf.Ticker(ticker, session=_yf_session()
                                        ).history(period=period),
                      tries=2, base_wait=1.0)
        if h is None or getattr(h, "empty", True) or "Close" not in h:
            return None
        c = h["Close"].dropna()
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        return c
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fred(series_id: str):
    """FRED CSV endpoint'i — API anahtarı gerektirmez (araştırma F).
    BAMLH0A0HYM2 = ICE BofA US High Yield OAS (günlük, 1996'dan).
    Hata → None (ekran HYG/IEF proxy'sine düşer)."""
    try:
        import io
        import requests
        r = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
            + series_id, timeout=10,
            headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[1] < 2:
            return None
        s = pd.to_numeric(df.iloc[:, 1].replace(".", np.nan),
                          errors="coerce")
        s.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        s = s[s.index.notna()].dropna()
        return s if len(s) else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# KRİPTO BAĞLAMI — SPOT al → NET HEDEF → STOP modülü (kripto araştırması)
# Doktrin hisse tarafıyla aynı: yön tahmini YOK; taban oranı + Wilson GA;
# çekirdek sinyal = trend/momentum (Liu-Tsyvinski 2021 RFS; Detzel vd.
# 2021 FM) + ATR stop (Kaminski-Lo 2014) + vol-ölçekli boyut (Man 2018).
# On-chain bantlar (MVRV-Z, Mayer) yalnız UÇ-bölge bağlamı; F&G ve
# funding yalnız BAYRAK (öngörü kanıtı yok) — çarpana girmezler.
# Veri tamamen ücretsiz: Binance public, CoinMetrics Community (CC BY-NC),
# alternative.me. Her "olasılık" TARİHSEL TABAN ORANIDIR, tahmin değil.
# ═══════════════════════════════════════════════════════════════════════
KRIPTO_UFUK_GUN = 90
KRIPTO_REDDEDILENLER = [
    "Stock-to-Flow / S2FX — sahte regresyon + tautoloji; 2022'de model "
    "kırıldı (Kripfganz kointegrasyon; Emblow; Buterin 'sahte kesinlik')",
    "Rainbow chart — log eğriye keyfi renk bandı; eğri uydurma",
    "Halving TARİHİ ile deterministik zamanlama — bilinen olay, "
    "fiyatlanır; sinyal kanıtı yok",
    "Chart pattern hedefleri (measured move, OBO, üçgen) — kripto "
    "kanıtı yok; Lo vd. (2000) hissede bile maliyet-sonrası belirsiz",
    "Elliott dalgaları — öznel, yanlışlanamaz",
    "Ay/astro takvimleri — kanıt yok",
    "Sosyal-hacim pump sinyalleri — manipülasyona açık",
    "'Altseason index' / dominance dönüş zamanlaması — kanıt yetersiz",
]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_kripto(sembol: str = "BTCUSDT", gunler: int = 1500):
    """Günlük OHLCV — 1) Binance spot klines (anahtarsız, 2×1000 mum
    sayfalı), 2) olmazsa yfinance {BAZ}-USD. Hata → None."""
    import requests
    try:
        satirlar, end = [], None
        for _ in range(2):
            p = {"symbol": sembol.upper(), "interval": "1d", "limit": 1000}
            if end:
                p["endTime"] = end
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params=p, timeout=15)
            r.raise_for_status()
            j = r.json()
            if not isinstance(j, list) or not j:
                break
            satirlar = j + satirlar
            end = j[0][0] - 1
            if len(j) < 1000 or len(satirlar) >= gunler:
                break
        if satirlar:
            df = pd.DataFrame(satirlar).iloc[:, [0, 1, 2, 3, 4, 5, 7]]
            df.columns = ["ts", "Open", "High", "Low", "Close",
                          "Volume", "QuoteVolume"]
            for k in df.columns[1:]:
                df[k] = pd.to_numeric(df[k], errors="coerce")
            df.index = pd.to_datetime(df["ts"], unit="ms")
            df = (df.drop(columns=["ts"]).dropna()
                    .loc[~df.index.duplicated()].sort_index().tail(gunler))
            if len(df) >= 260:
                return df
    except Exception:
        pass
    try:  # yfinance fallback (BTC-USD, ETH-USD, SOL-USD…)
        import yfinance as yf
        baz = sembol.upper()
        for ek in ("USDT", "USDC", "BUSD", "TRY", "USD"):
            baz = baz[:-len(ek)] if baz.endswith(ek) else baz
        t = yf.Ticker(f"{baz}-USD", session=_yf_session())
        h = _yf_retry(lambda: t.history(period="5y", auto_adjust=False))
        if h is not None and not h.empty:
            h = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
            h.index = pd.to_datetime(h.index).tz_localize(None)
            h["QuoteVolume"] = h["Close"] * h["Volume"]
            if len(h) >= 260:
                return h.tail(gunler)
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fg_kripto():
    """alternative.me Fear&Greed (anahtarsız). Öngörü kanıtı YOK —
    yalnız uç-bayrak (VAR çalışması: F&G tepkiseldir)."""
    import requests
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1",
                         timeout=10)
        r.raise_for_status()
        d = (r.json().get("data") or [{}])[0]
        return {"deger": int(d.get("value")),
                "etiket": d.get("value_classification")}
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_funding_kripto(sembol: str = "BTCUSDT"):
    """Binance USD-M anlık funding (anahtarsız). Pozisyonlanma BAYRAĞI —
    tahmin değil (carry Sharpe'ı 2025'te negatife döndü)."""
    import requests
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                         params={"symbol": sembol.upper()}, timeout=10)
        r.raise_for_status()
        j = r.json()
        f = float(j.get("lastFundingRate"))
        return {"funding": f, "sembol": sembol.upper()}
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_btc_onchain():
    """CoinMetrics Community (CC BY-NC 4.0; kişisel kullanım): BTC
    market cap + realized cap → MVRV ve MVRV-Z serisi. Hata → None."""
    import requests
    try:
        url = ("https://community-api.coinmetrics.io/v4/timeseries/"
               "asset-metrics")
        params = {"assets": "btc",
                  "metrics": "CapMrktCurUSD,CapRealUSD",
                  "frequency": "1d", "page_size": 10000,
                  "start_time": "2013-01-01"}
        veriler, tur = [], 0
        while tur < 4:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            j = r.json()
            veriler += j.get("data") or []
            nxt = j.get("next_page_url")
            if not nxt:
                break
            url, params, tur = nxt, {}, tur + 1
        if not veriler:
            return None
        df = pd.DataFrame(veriler)
        df.index = pd.to_datetime(df["time"]).dt.tz_localize(None)
        mc = pd.to_numeric(df["CapMrktCurUSD"], errors="coerce")
        rc = pd.to_numeric(df["CapRealUSD"], errors="coerce")
        o = pd.DataFrame({"mcap": mc, "rcap": rc}).dropna()
        o["mvrv"] = o["mcap"] / o["rcap"]
        o["mvrv_z"] = (o["mcap"] - o["rcap"]) / \
            o["mcap"].expanding(min_periods=365).std()
        return o
    except Exception:
        return None


def _kripto_atr(df, n=20):
    """ATR — Wilder TR'nin n-günlük ortalaması."""
    try:
        h, l, c = df["High"], df["Low"], df["Close"]
        co = c.shift(1)
        tr = pd.concat([h - l, (h - co).abs(), (l - co).abs()],
                       axis=1).max(axis=1)
        return tr.rolling(n).mean()
    except Exception:
        return None


def kripto_rejim(df, onchain=None, fg=None, funding=None):
    """Rejim + boyut çarpanı (0.70-1.30, hisse tarafıyla aynı kompozit
    mantık). Çarpana giren: trend, vol rejimi, uç-değerleme. Girmeyen
    (yalnız bayrak): F&G, funding — öngörü kanıtları yok."""
    try:
        c = df["Close"].dropna()
        if len(c) < 260:
            return None
        son = float(c.iloc[-1])
        ma200 = float(c.rolling(200).mean().iloc[-1])
        trend = son >= ma200
        mom12 = float(son / c.iloc[-85] - 1) if len(c) >= 85 else None
        mayer = son / ma200 if ma200 else None
        rv = c.pct_change().rolling(30).std() * np.sqrt(365)
        rv = rv.dropna()
        vol30 = float(rv.iloc[-1]) if len(rv) else None
        vol_pct = float((rv < vol30).mean()) if vol30 is not None else None
        mvrv_z = None
        if onchain is not None and len(onchain):
            oc = onchain[onchain.index <= c.index[-1]]
            if len(oc):
                mvrv_z = float(oc["mvrv_z"].iloc[-1])

        skorlar, gerekce = {}, []
        skorlar["trend"] = 0.25 if trend else -1.0
        gerekce.append("trend: fiyat 200g MA "
                       + ("ÜSTÜNDE → +0.25" if trend else "ALTINDA → −1.0")
                       + " (Detzel vd. 2021; Liu-Tsyvinski 2021)")
        if vol_pct is not None:
            v = (-1.0 if vol_pct >= 0.8 else -0.5 if vol_pct >= 0.6
                 else 0.5 if vol_pct <= 0.2 else 0.0)
            skorlar["vol"] = v
            gerekce.append(f"volatilite: 30g gerçekleşen vol kendi "
                           f"tarihinin %{vol_pct*100:.0f}. yüzdeliği → "
                           f"{v:+.1f} (Man 2018 vol hedefleme)")
        asiri_ust = ((mvrv_z is not None and mvrv_z > 6)
                     or (mayer is not None and mayer > 2.4))
        dip_bolge = ((mvrv_z is not None and mvrv_z < 0)
                     or (mayer is not None and mayer < 0.8))
        if asiri_ust:
            skorlar["degerleme"] = -1.0
            gerekce.append("değerleme: MVRV-Z>6 / Mayer>2.4 — tarihsel "
                           "aşırı bölge → −1.0 (topluluk sezgisi, "
                           "döngüde zayıflıyor)")
        elif dip_bolge:
            skorlar["degerleme"] = 0.5
            gerekce.append("değerleme: MVRV-Z<0 / Mayer<0.8 — tarihsel "
                           "dip bölgesi → +0.5 (zayıf kanıt etiketi)")
        else:
            skorlar["degerleme"] = 0.0
            gerekce.append("değerleme: uç bölgede değil → 0.0")
        ort = float(np.mean(list(skorlar.values())))
        carpan = float(max(0.70, min(1.30,
                                     1.0 + 0.30 * max(-1.0, min(1.0, ort)))))

        bayraklar = []
        if fg and fg.get("deger") is not None:
            if fg["deger"] >= 90:
                bayraklar.append(f"🚩 F&G {fg['deger']} (aşırı açgözlü) — "
                                 f"tepkisel gösterge, öngörü kanıtı yok")
            elif fg["deger"] <= 10:
                bayraklar.append(f"🚩 F&G {fg['deger']} (aşırı korku) — "
                                 f"tepkisel gösterge, öngörü kanıtı yok")
        if funding and funding.get("funding") is not None:
            fr = funding["funding"]
            if fr >= 0.0005:
                bayraklar.append(f"🚩 Funding %{fr*100:.3f}/8s — kalabalık "
                                 f"LONG; kısa vadeli sarsıntı riski")
            elif fr <= -0.0003:
                bayraklar.append(f"🚩 Funding %{fr*100:.3f}/8s negatif — "
                                 f"kalabalık SHORT")
        return {"son": son, "ma200": ma200, "trend": trend,
                "mom12": mom12, "mayer": (round(mayer, 2) if mayer else None),
                "mvrv_z": (round(mvrv_z, 2) if mvrv_z is not None else None),
                "vol30": vol30, "vol_pct": vol_pct,
                "carpan": round(carpan, 2), "skorlar": skorlar,
                "gerekce": gerekce, "bayraklar": bayraklar}
    except Exception:
        return None


def _ilk_gecis(df, maske, stop_pct, hedef_pct, ufuk=KRIPTO_UFUK_GUN,
               min_n=25):
    """AMPİRİK ilk-geçiş taban oranı: benzer-rejim günlerinden girilse,
    ufuk içinde fiyat +hedef'e mi −stop'a mı önce ulaşmış? Günlük H/L
    ile; AYNI GÜN ikisi de vurulursa MUHAFAZAKÂR sayım = stop. Örtüşme
    düzeltmesi: etkin N ≈ N / ortalama sonuç süresi. Tahmin DEĞİL."""
    try:
        H = df["High"].values.astype(float)
        L = df["Low"].values.astype(float)
        C = df["Close"].values.astype(float)
        idx = np.where(np.asarray(maske, dtype=bool))[0]
        hedef = stopv = zaman = 0
        sureler = []
        for i in idx:
            if i + 1 + ufuk > len(C):
                continue
            g = C[i]
            hs, ss = g * (1 + hedef_pct), g * (1 - stop_pct)
            sonuc = None
            for j in range(i + 1, i + 1 + ufuk):
                alt, ust = L[j] <= ss, H[j] >= hs
                if alt:                       # aynı gün ikisi → stop
                    sonuc = ("stop", j - i)
                    break
                if ust:
                    sonuc = ("hedef", j - i)
                    break
            if sonuc is None:
                zaman += 1
                sureler.append(ufuk)
            elif sonuc[0] == "hedef":
                hedef += 1
                sureler.append(sonuc[1])
            else:
                stopv += 1
                sureler.append(sonuc[1])
        n = hedef + stopv + zaman
        if n < min_n:
            return {"yetersiz": True, "n": n}
        ort_sure = float(np.mean(sureler)) if sureler else float(ufuk)
        etkin = max(1, int(n / max(1.0, ort_sure)))
        p = hedef / n
        lo, hi = _wilson(int(round(p * etkin)), etkin)
        return {"yetersiz": False, "n": n, "etkin": etkin,
                "oran": round(p, 3), "stop_oran": round(stopv / n, 3),
                "zaman_oran": round(zaman / n, 3),
                "ga": (lo, hi), "ort_sure": round(ort_sure, 1)}
    except Exception:
        return {"yetersiz": True, "n": 0}


def kripto_plan(df, rejim, k_atr=2.5, ufuk=KRIPTO_UFUK_GUN,
                rr_grid=(1.5, 2.0, 3.0), ucret=0.002, alt_mi=False):
    """AL → NET HEDEF → STOP planı. Stop = k×ATR20 ile 20g yapısal dip
    (×0.995) — hangisi DAHA GENİŞSE (whipsaw azaltma), max-zarar tavanı
    (BTC %15 / alt %25) aşılırsa ATR stopa dönülür. Hedefler R-katı;
    her hedefe o coin'in KENDİ geçmişinden rejim-koşullu ilk-geçiş taban
    oranı iliştirilir; ücret düzeltmeli net R:R gösterilir."""
    try:
        c = df["Close"].dropna()
        atr = _kripto_atr(df)
        if rejim is None or atr is None or atr.dropna().empty:
            return None
        giris = float(c.iloc[-1])
        a = float(atr.iloc[-1])
        stop_atr = giris - k_atr * a
        yapisal = float(df["Low"].iloc[-20:].min()) * 0.995
        stop = min(stop_atr, yapisal)
        kaynak = "yapısal 20g dip" if yapisal < stop_atr else \
            f"{k_atr:.1f}×ATR20"
        tavan = 0.25 if alt_mi else 0.15
        mesafe = 1 - stop / giris
        tavan_notu = None
        if mesafe > tavan:
            stop, kaynak = stop_atr, f"{k_atr:.1f}×ATR20 (tavan)"
            mesafe = 1 - stop / giris
            tavan_notu = (f"Yapısal stop max-zarar tavanını "
                          f"(%{tavan*100:.0f}) aşıyordu — ATR stopa "
                          f"dönüldü.")
        if mesafe > tavan or mesafe <= 0:
            return {"gecersiz": True,
                    "not": (f"Stop mesafesi %{mesafe*100:.1f} — "
                            f"tavanın (%{tavan*100:.0f}) üstünde. Bu "
                            f"volatilitede işlem KÜÇÜLT veya ATLA.")}
        # rejim maskesi: bugünle aynı trend + momentum işareti
        ma200 = c.rolling(200).mean()
        mom = c / c.shift(84) - 1
        maske = ((c >= ma200) == bool(rejim["trend"]))
        if rejim.get("mom12") is not None:
            maske &= ((mom > 0) == (rejim["mom12"] > 0))
        maske &= ma200.notna() & mom.notna()
        satirlar = []
        for rr in rr_grid:
            hp = rr * mesafe
            fg_ = _ilk_gecis(df, maske, mesafe, hp, ufuk)
            satirlar.append({
                "rr": rr, "hedef": round(giris * (1 + hp), 6),
                "hedef_pct": round(hp, 4),
                "net_rr": round((hp - ucret) / (mesafe + ucret), 2),
                "gecis": fg_})
        return {"gecersiz": False, "giris": giris, "stop": round(stop, 6),
                "stop_pct": round(mesafe, 4), "kaynak": kaynak,
                "atr_pct": round(a / giris, 4), "satirlar": satirlar,
                "tavan_notu": tavan_notu, "ucret": ucret, "ufuk": ufuk,
                "sinir": (
                    "Oranlar TARİHSEL taban oranıdır (tahmin değil); "
                    "aynı-gün çift-vuruş STOP sayılır (muhafazakâr); "
                    "gözlemler örtüşür — etkin N ona göre küçültüldü; "
                    "fitil/slippage stopu aşabilir (10 Eki 2025 örneği).")}
    except Exception:
        return None


# ── ⚡ GÜNLÜK İŞLEM MODU (kısa vade, 1h) — day-trade araştırması ──────
# Dürüst temel: 300+ gün ısrar eden day-trader'ların %97'si zararda
# (Chague-De-Losso-Giovannetti 2020); kalıcı kârlı oran ~%5 (Barber vd.
# 2020); CFD hesaplarının %74-89'u kayıpta (ESMA 2018). Bu modun işi
# kâr vaadi değil, HASAR SINIRLAMA: zorunlu uyarı + günde 3 plan +
# 2 zararda gün kapanışı (Coval-Shumway 2005 tilt; Locke-Mann 2005) +
# gürültü-tabanı stop + fee-başabaş kontrolü.
GI_UFUK_BAR = 48          # zaman stopu: 48 saat içinde dolmayan işlem kesilir
GI_SEANS_UTC = (13, 21)   # likit pencere ~13-21 UTC (Brauneis vd. 2025)
GI_REDDEDILENLER = [
    "<%0.5 hedefli scalp — fee matematiği yapısal kaybettirir "
    "(%0.2 gidiş-dönüşte başabaş kazanma oranı ~%70)",
    "Tek başına RSI/MACD/Bollinger girişleri — 7.846 intraday kuralın "
    "hiçbiri veri-tarama düzeltmesi sonrası kârlı değil "
    "(Marshall-Cahan-Cahan 2008)",
    "Sinyal grupları / pump kanalları — dikkat-kaynaklı alım 20 günde "
    "−%4.7 (Barber vd. 2022)",
    "Martingale / grid bot — iflas olasılığını büyütür",
    "Düşen bıçağı yakalama (counter-trend) — zarar sonrası risk artışı "
    "geri teper (Coval-Shumway 2005)",
    "Haber-spike kovalama ve intikam işlemi (revenge trade)",
    "Düşük likidite altcoin scalp — 10 Eki 2025'te spread 0.02→26 bps, "
    "derinlik %98 düştü",
]


@st.cache_data(ttl=600, show_spinner=False)
def fetch_kripto_saatlik(sembol: str = "BTCUSDT", bar: int = 12000):
    """Binance spot 1h klines (anahtarsız), sayfalı ~12×1000 mum
    (~500 gün). SON (henüz kapanmamış) mum ATILIR. Hata → None."""
    import requests
    try:
        satirlar, end = [], None
        for _ in range(12):
            p = {"symbol": sembol.upper(), "interval": "1h",
                 "limit": 1000}
            if end:
                p["endTime"] = end
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params=p, timeout=15)
            r.raise_for_status()
            j = r.json()
            if not isinstance(j, list) or not j:
                break
            satirlar = j + satirlar
            end = j[0][0] - 1
            if len(j) < 1000 or len(satirlar) >= bar:
                break
        if not satirlar:
            return None
        df = pd.DataFrame(satirlar).iloc[:, [0, 1, 2, 3, 4, 5, 7]]
        df.columns = ["ts", "Open", "High", "Low", "Close",
                      "Volume", "QuoteVolume"]
        for k in df.columns[1:]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        df.index = pd.to_datetime(df["ts"], unit="ms")
        df = (df.drop(columns=["ts"]).dropna()
                .loc[~df.index.duplicated()].sort_index())
        df = df.iloc[:-1]                 # kapanmamış mum yok
        return df.tail(bar) if len(df) >= 400 else None
    except Exception:
        return None


def _gi_con():
    import sqlite3
    con = sqlite3.connect("gunluk_disiplin.db")
    con.execute("CREATE TABLE IF NOT EXISTS g("
                "tarih TEXT PRIMARY KEY, plan INTEGER DEFAULT 0, "
                "zarar INTEGER DEFAULT 0, kar INTEGER DEFAULT 0)")
    con.commit()
    return con


def _gi_gun():
    return datetime.now().strftime("%Y-%m-%d")


def _gi_oku(con, gun):
    r = con.execute("SELECT plan, zarar, kar FROM g WHERE tarih=?",
                    (gun,)).fetchone()
    return {"plan": r[0], "zarar": r[1], "kar": r[2]} if r else \
        {"plan": 0, "zarar": 0, "kar": 0}


def _gi_art(con, gun, alan):
    con.execute("INSERT INTO g(tarih) VALUES(?) "
                "ON CONFLICT(tarih) DO NOTHING", (gun,))
    con.execute(f"UPDATE g SET {alan}={alan}+1 WHERE tarih=?", (gun,))
    con.commit()


def gerekli_kazanma(hedef_pct, stop_pct, ucret=0.002):
    """Fee'li başabaş kazanma oranı: p* = (S+f)/(T+S).
    T=%2, S=%1, f=%0.2 → %40; T=S=%0.5 → %70 (scalp'in mezarı)."""
    try:
        return (stop_pct + ucret) / (hedef_pct + stop_pct)
    except Exception:
        return None


def _gi_swing_dip(df1h, pencere=48, kanat=3):
    """Objektif 1h swing-low: son `pencere` bar içinde, iki yanındaki
    `kanat` bardan düşük kalan en yakın pivot dip. Yoksa None."""
    try:
        L = df1h["Low"].values.astype(float)
        n = len(L)
        for i in range(n - 1 - kanat, max(kanat, n - pencere) - 1, -1):
            sol = L[i - kanat:i]
            sag = L[i + 1:i + 1 + kanat]
            if len(sag) < kanat:
                continue
            if L[i] < sol.min() and L[i] <= sag.min():
                return float(L[i])
        return None
    except Exception:
        return None


def gunluk_setup(df1h, htf_yukari):
    """Pullback-reclaim dedektörü (tek savunulabilir beginner girişi:
    HTF trend + saatlik geri çekilme sonrası toparlanma). Koşullar tek
    tek raporlanır — 'setup yok' da bir cevaptır."""
    try:
        atr = _kripto_atr(df1h, 14)
        c, h = df1h["Close"], df1h["High"]
        a = float(atr.iloc[-1])
        son = float(c.iloc[-1])
        tepe24 = float(h.iloc[-24:].max())
        derinlik = (tepe24 - float(df1h["Low"].iloc[-2])) / a if a else None
        reclaim = bool(son > float(h.iloc[-2]))
        import time as _t
        saat = _t.gmtime().tm_hour
        seans = GI_SEANS_UTC[0] <= saat < GI_SEANS_UTC[1]
        kosullar = [
            ("Günlük trend yukarı (200g MA üstü + momentum +)",
             bool(htf_yukari)),
            (f"Likit saat ({GI_SEANS_UTC[0]}-{GI_SEANS_UTC[1]} UTC = "
             f"{GI_SEANS_UTC[0]+3}:00-{GI_SEANS_UTC[1]+3}:00 TRT)", seans),
            ("Geri çekilme var (24s tepeden ≥0.8×ATR, ≤3×ATR)",
             bool(derinlik is not None and 0.8 <= derinlik <= 3.0)),
            ("Toparlanma (son kapanan 1s mum, önceki mumun tepesini "
             "aştı)", reclaim),
        ]
        return {"aktif": all(ok for _, ok in kosullar),
                "kosullar": kosullar, "atr1h": a,
                "atr1h_pct": (a / son if son else None),
                "derinlik_atr": (round(derinlik, 2)
                                 if derinlik is not None else None),
                "tepe24": tepe24, "son": son}
    except Exception:
        return None


def gunluk_plan(df1h, k_atr=1.75, ufuk_bar=GI_UFUK_BAR, alt_mi=False,
                ucret=0.002, rr_grid=(1.0, 1.5, 2.0)):
    """1h AL → HEDEF → STOP planı. Stop = max-mesafe(k×ATR(14,1h),
    swing-low − 0.25×ATR tamponu); GÜRÜLTÜ TABANI: stop ≥ 3×(2f+spread)
    ≈ %0.6; tavan BTC %6 / alt %10. Her hedefe (a) 1h ilk-geçiş taban
    oranı, (b) fee'li GEREKEN kazanma oranı iliştirilir; taban oranı
    gerekenin altındaysa satır ✖ işaretlenir."""
    try:
        c = df1h["Close"].dropna()
        atr = _kripto_atr(df1h, 14)
        if atr is None or atr.dropna().empty or len(c) < 400:
            return None
        giris = float(c.iloc[-1])
        a = float(atr.iloc[-1])
        stop_atr = giris - k_atr * a
        dip = _gi_swing_dip(df1h)
        stop_yapisal = (dip - 0.25 * a) if dip is not None else None
        adaylar = [s for s in (stop_atr, stop_yapisal) if s is not None]
        stop = min(adaylar)
        kaynak = ("swing-dip − 0.25×ATR tamponu"
                  if stop == stop_yapisal else f"{k_atr:.2f}×ATR(14,1h)")
        mesafe = 1 - stop / giris
        taban = 0.006                      # gürültü tabanı ≈ 3×(2f+spread)
        tavan = 0.10 if alt_mi else 0.06
        if mesafe < taban:
            stop = giris * (1 - taban)
            mesafe, kaynak = taban, "gürültü tabanı (%0.6)"
        if mesafe > tavan:
            stop = stop_atr
            mesafe = 1 - stop / giris
            kaynak = f"{k_atr:.2f}×ATR (tavan)"
        if mesafe > tavan or mesafe <= 0:
            return {"gecersiz": True,
                    "not": (f"Stop mesafesi %{mesafe*100:.1f} — günlük "
                            f"işlem tavanının (%{tavan*100:.0f}) üstünde."
                            f" Bu volatilitede GÜNLÜK işlem açma; swing "
                            f"moduna bak ya da bekle.")}
        # rejim maskesi: 1 haftalık MA üstü + benzer geri çekilme derinliği
        ma168 = c.rolling(168).mean()
        tepe24r = df1h["High"].rolling(24).max()
        der = (tepe24r - c) / atr
        maske = (c > ma168) & der.between(0.5, 3.0) & atr.notna() \
            & ma168.notna()
        satirlar = []
        for rr in rr_grid:
            hp = rr * mesafe
            fg_ = _ilk_gecis(df1h, maske, mesafe, hp, ufuk=ufuk_bar)
            pg = gerekli_kazanma(hp, mesafe, ucret)
            uygun = (None if fg_.get("yetersiz")
                     else bool(fg_["oran"] >= pg))
            satirlar.append({"rr": rr,
                             "hedef": round(giris * (1 + hp), 6),
                             "hedef_pct": round(hp, 4),
                             "net_rr": round((hp - ucret) / (mesafe + ucret), 2),
                             "gerekli": round(pg, 3), "gecis": fg_,
                             "uygun": uygun})
        return {"gecersiz": False, "giris": giris,
                "stop": round(stop, 6), "stop_pct": round(mesafe, 4),
                "kaynak": kaynak, "atr_pct": round(a / giris, 4),
                "satirlar": satirlar, "ufuk_bar": ufuk_bar,
                "sinir": (
                    f"Oranlar 1s barlarla TARİHSEL taban oranıdır; aynı "
                    f"barda çift-vuruş STOP sayılır (1s'de belirsizlik "
                    f"günlükten büyüktür — muhafazakâr); gözlemler "
                    f"örtüşür, etkin N küçültüldü. {ufuk_bar} saatte "
                    f"ikisi de dolmazsa ZAMAN STOPU: çık.")}
    except Exception:
        return None


def render_gunluk_islem():
    st.markdown("# ⚡ Günlük İşlem (Kısa Vade) — 1 Saatlik")
    # ── ZORUNLU UYARI KAPISI ─────────────────────────────────────────
    if not st.session_state.get("gi_onay"):
        st.error(
            "🛑 **ÖNCE İSTATİSTİK, SONRA İŞLEM.** Günlük işlemde 300+ "
            "gün ısrar edenlerin **%97'si zararda** bitirdi (Brezilya, "
            "19.646 kişi; Chague vd. 2020). Kalıcı kârlı day-trader "
            "oranı **~%5** (Tayvan, Barber vd. 2020). Kaldıraçlı CFD "
            "hesaplarının **%74-89'u** kayıpta (ESMA 2018). Deneyimle "
            "düzelme KANITLANAMADI. Bu mod seni zengin etmek için "
            "değil, HASARI SINIRLAMAK için tasarlandı: günde en fazla "
            "3 plan, 2 zararda gün kapanır, fee matematiği her satırda.")
        if st.checkbox("Bu istatistikleri okudum; beklenen değeri "
                       "negatif bir faaliyet olduğunu anlıyorum ve "
                       "kaybetmeyi göze alabileceğim parayla, küçük "
                       "risk yüzdesiyle devam etmek istiyorum.",
                       key="gi_onay_kutu"):
            st.session_state["gi_onay"] = True
            st.rerun()
        return

    # ── DİSİPLİN KİLİTLERİ (sqlite — uygulama kapansa da hatırlar) ──
    con = _gi_con()
    gun = _gi_gun()
    d = _gi_oku(con, gun)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Bugünkü plan", f"{d['plan']}/3")
    k2.metric("Zararla kapanan", f"{d['zarar']}/2")
    k3.metric("Kârla kapanan", d["kar"])
    import time as _t
    saat_utc = _t.gmtime().tm_hour
    seans = GI_SEANS_UTC[0] <= saat_utc < GI_SEANS_UTC[1]
    k4.metric("Seans", "LİKİT ✓" if seans else "İNCE",
              f"{GI_SEANS_UTC[0]+3}:00-{GI_SEANS_UTC[1]+3}:00 TRT",
              delta_color="off")
    b1, b2 = st.columns(2)
    if b1.button("✅ Son işlemim KÂRLA kapandı", width="stretch"):
        _gi_art(con, gun, "kar")
        st.rerun()
    if b2.button("❌ Son işlemim ZARARLA kapandı", width="stretch"):
        _gi_art(con, gun, "zarar")
        st.rerun()
    if d["zarar"] >= 2:
        st.error(
            "⛔ **BUGÜN PLAN YAPMIYORUM — GÜN KAPANDI.** Sebep: bugün "
            "2 işlem zararla kapandı (cool-off). Kanıt: zarar sonrası "
            "işlemciler ~%16 daha fazla risk alıyor ve bu işlemler "
            "geri tepiyor (Coval-Shumway 2005, CBOT); en kötü "
            "işlemciler kaybedeni en uzun tutuyor (Locke-Mann 2005). "
            "Yarın likit saatte tekrar bak. Bugün ekranı kapat.")
        st.caption("Kural değişmez — bu kilit senin sermayeni koruyor.")
        return
    if d["plan"] >= 3:
        st.error("⛔ **Bugünlük 3 plan hakkı doldu.** Aşırı işlem = "
                 "düşük getiri (Barber-Odean 2000). Yarın devam.")
        return
    if not seans:
        st.warning(f"🌙 Şu an likit pencere dışındasın "
                   f"({saat_utc:02d}:00 UTC). Spread geniş, dolum "
                   f"kötü — varsayılan öneri: {GI_SEANS_UTC[0]+3}:00 "
                   f"TRT sonrası bak. (Yine de plan kurabilirsin; "
                   f"karar senin.)")

    # ── GİRDİLER ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    sembol = c1.text_input("Parite (Binance spot)", value="BTCUSDT",
                           key="gi_sembol").strip().upper()
    sermaye = c2.number_input("Sermaye ($)", 100.0, 1e9, 1000.0, 100.0,
                              key="gi_sermaye")
    risk = c3.slider("İşlem riski %", 0.10, 0.50, 0.35, 0.05,
                     key="gi_risk",
                     help="Günlük işlemde swing'den DÜŞÜK tutulur — "
                          "taban oranlar kötü. Stop yerse kaybın bu.")
    k_atr = c4.slider("Stop çarpanı (×ATR 1s)", 1.5, 2.5, 1.75, 0.05,
                      key="gi_katr")
    if not st.button("⚡ Günlük Planı Kur", type="primary",
                     width="stretch"):
        return

    alt_mi = not sembol.startswith("BTC")
    with st.spinner("1 saatlik veri çekiliyor (~500 gün)…"):
        df1h = fetch_kripto_saatlik(sembol)
        dfg = fetch_kripto(sembol)
    if df1h is None or dfg is None:
        st.error(f"{sembol} verisi alınamadı (1s için Binance'te "
                 f"listeli olmalı).")
        return
    rejg = kripto_rejim(dfg)
    htf_yukari = bool(rejg and rejg["trend"]
                      and (rejg.get("mom12") or 0) > 0)
    if alt_mi:
        btc = fetch_kripto("BTCUSDT")
        btc_rej = kripto_rejim(btc) if btc is not None else None
        if btc_rej and not btc_rej["trend"]:
            st.error("🚪 BTC 200g MA altında — alt-coin GÜNLÜK işlemi "
                     "bu rejimde önerilmez (çöküşte korelasyon → 1).")

    setup = gunluk_setup(df1h, htf_yukari)
    pl = gunluk_plan(df1h, k_atr=k_atr, alt_mi=alt_mi)
    if setup is None or pl is None:
        st.error("Hesaplanamadı (en az ~400 kapanmış 1s mum gerek).")
        return
    _gi_art(con, gun, "plan")

    # ── SETUP DURUMU ────────────────────────────────────────────────
    st.markdown("#### 🎛️ Setup: HTF trend + saatlik geri çekilme + "
                "toparlanma")
    for ad, ok in setup["kosullar"]:
        (st.success if ok else st.warning)(("✓ " if ok else "✗ ") + ad)
    if setup["aktif"]:
        st.success("🟢 **SETUP AKTİF** — koşulların hepsi sağlandı. "
                   "Aşağıdaki plan geçerli.")
    else:
        st.info("⚪ **SETUP YOK** — 'işlem açmamak' da bir karardır ve "
                "çoğu gün DOĞRU karardır. Plan yine de aşağıda; ama "
                "koşullar sağlanmadan girmek istatistiği daha da "
                "aleyhe çevirir.")

    if pl.get("gecersiz"):
        st.error("🛑 " + pl["not"])
        return

    # ── PLAN KARTI ──────────────────────────────────────────────────
    boyut_pct = min((risk / 100.0) / pl["stop_pct"], 0.25)
    tutar = sermaye * boyut_pct
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Giriş", f"{pl['giris']:,.6g}")
    p2.metric("Stop", f"{pl['stop']:,.6g}",
              f"−%{pl['stop_pct']*100:.2f} · {pl['kaynak']}",
              delta_color="off")
    p3.metric("ATR(14,1s)", f"%{pl['atr_pct']*100:.2f}")
    p4.metric("Pozisyon", f"${tutar:,.0f}",
              f"sermayenin %{boyut_pct*100:.1f}", delta_color="off")
    tablo = []
    for s in pl["satirlar"]:
        g = s["gecis"]
        if g.get("yetersiz"):
            oran, ga, isaret = f"veri az (N={g.get('n', 0)})", "—", "—"
        else:
            oran = f"%{g['oran']*100:.0f}"
            ga = (f"%{g['ga'][0]*100:.0f}–%{g['ga'][1]*100:.0f} "
                  f"(etkin N≈{g['etkin']})")
            isaret = "✓" if s["uygun"] else "✖"
        tablo.append({"Hedef": f"{s['rr']:.1f}R",
                      "Fiyat": f"{s['hedef']:,.6g}",
                      "Getiri": f"+%{s['hedef_pct']*100:.2f}",
                      "Gereken kazanma (fee'li)": f"%{s['gerekli']*100:.0f}",
                      "Tarihsel ilk-geçiş": oran,
                      "Wilson GA": ga,
                      "Fee testi": isaret})
    st.dataframe(pd.DataFrame(tablo), hide_index=True, width="stretch")
    if any(s["uygun"] is False for s in pl["satirlar"]):
        st.error("🔴 ✖ işaretli hedeflerde tarihsel oran, fee'li "
                 "GEREKEN kazanma oranının ALTINDA — o hedefle işlem "
                 "matematiksel olarak aleyhine. ✓ olmayan plan AÇMA.")
    st.warning("⚠️ " + pl["sinir"])

    # ── SALAK-GEÇİRMEZ KART ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**🗣️ TEK NEFESTE PLAN:**")
        st.markdown(f"- **NEREDEN GİRİLİR:** Setup AKTİF ise "
                    f"~{pl['giris']:,.6g} civarı. Değilse GİRME.")
        st.markdown(f"- **STOP NEREYE:** {pl['stop']:,.6g} "
                    f"(−%{pl['stop_pct']*100:.2f}) — dokunursa çık, "
                    f"pazarlık yok.")
        st.markdown(f"- **HEDEF:** ✓ işaretli en düşük R hedefi; "
                    f"dolunca SAT, açgözlülük yok.")
        st.markdown(f"- **ZAMAN STOPU:** {pl['ufuk_bar']} saatte ikisi "
                    f"de dolmadıysa çık.")
        st.markdown("- **BUGÜN AÇMA EĞER:** 2 zarar olduysa · 3 plan "
                    "dolduysa · likit saat dışıysa · setup yoksa.")
        st.markdown("- **NE DEMEK DEĞİL:** 'Kesin kazanç' değil; "
                    "'zengin olursun' değil. Oranlar geçmiş taban "
                    "oranı — gelecek garantisi yok.")
    with st.expander("🚫 Günlük modda KASITLI OLARAK kullanılmayanlar"):
        for x in GI_REDDEDILENLER:
            st.markdown(f"- ✖ {x}")
    st.caption("Bu bir disiplin çerçevesidir; yatırım tavsiyesi "
               "değildir. Kaybetmeyi göze alamayacağın parayla işlem "
               "yapma.")


def render_kripto():
    st.markdown("# 🪙 Kripto Bağlamı — Spot Al → Hedef → Stop")
    _mk = st.radio("İşlem tarzı",
                   ["🌊 Swing (≈90 gün)",
                    "⚡ Günlük İşlem (24-72 saat)"],
                   horizontal=True, key="kr_mod",
                   help="Günlük mod: 1 saatlik mumlar, zorunlu "
                        "istatistik uyarısı, günde 3 plan + 2 "
                        "zararda gün kilidi, fee-başabaş testi.")
    if _mk.startswith("⚡"):
        render_gunluk_islem()
        return
    st.error(
        "⚖️ **ÖNCE BUNU OKU.** Bu ekran yön TAHMİN ETMEZ. Yaptığı şey: "
        "trend/vol rejimini ölçmek (kanıt: Liu-Tsyvinski 2021; Detzel "
        "2021; Man 2018), ATR'ye göre stop + R-katı hedef kurmak ve her "
        "hedefe coin'in KENDİ geçmişinden 'benzer rejimde +hedefe mi "
        "−stopa mı önce ulaşılmış' TABAN ORANINI iliştirmek. Kripto "
        "kanıtı incedir ve bozulur (ETF çağı) — belirsizlik her satırda "
        "görünür tutulur.")
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    sembol = c1.text_input("Parite (Binance spot)", value="BTCUSDT",
                           key="kr_sembol",
                           help="Örn. BTCUSDT, ETHUSDT, SOLUSDT. "
                                "Binance'te yoksa yfinance {BAZ}-USD "
                                "denenir.").strip().upper()
    sermaye = c2.number_input("Sermaye ($)", 100.0, 1e9, 1000.0, 100.0)
    risk = c3.slider("İşlem riski %", 0.10, 1.50, 0.75, 0.05,
                     help="Stop yerse kaybedilecek sermaye yüzdesi. "
                          "Kanıt çerçevesi: sabit-kesirli risk; BTC "
                          "için ~%0.5-1.0, ALTCOIN için %0.25-0.5 "
                          "önerilir (şişman kuyruk).")
    k_atr = c4.slider("Stop çarpanı (×ATR20)", 2.0, 3.5, 2.5, 0.1,
                      help="Kaminski-Lo (2014): stop momentum rejiminde "
                           "değer katar. 2-3× pratik standart; altta "
                           "3× öner.")
    if not st.button("🪙 Planı Kur", type="primary", width="stretch"):
        return
    alt_mi = not sembol.startswith("BTC")
    with st.spinner("Kripto verisi çekiliyor (Binance/CoinMetrics)…"):
        df = fetch_kripto(sembol)
        btc = df if not alt_mi else fetch_kripto("BTCUSDT")
        onchain = fetch_btc_onchain() if not alt_mi else None
        fg = fetch_fg_kripto()
        fund = fetch_funding_kripto(sembol)
    if df is None:
        st.error(f"{sembol} verisi alınamadı — pariteyi kontrol et "
                 f"(örn. BTCUSDT).")
        return
    rejim = kripto_rejim(df, onchain, fg, fund)
    if rejim is None:
        st.error("Rejim hesaplanamadı (en az ~1 yıl günlük veri gerek).")
        return

    # ── ALT KAPILARI ─────────────────────────────────────────────────
    if alt_mi:
        st.markdown("#### 🚪 Altcoin kapıları")
        btc_rejim = kripto_rejim(btc) if btc is not None else None
        if btc_rejim and not btc_rejim["trend"]:
            st.error("🚪 **BTC-REJİM KAPISI KAPALI** — BTC 200g MA "
                     "altında. Çöküşte alt korelasyonu 1'e gider; alt "
                     "işlemi bu rejimde ÖNERİLMEZ (küçült/atla).")
        elif btc_rejim:
            st.success("🚪 BTC-rejim kapısı AÇIK (BTC trend yukarı).")
        qv20 = float(df["QuoteVolume"].iloc[-20:].mean()) \
            if "QuoteVolume" in df else None
        if qv20 is not None and qv20 < 10e6:
            st.error(f"🚪 **LİKİDİTE KAPISI KAPALI** — 20g ort. hacim "
                     f"${qv20/1e6:.1f}M < $10M. İnce defter: slippage "
                     f"ve fitil riski yüksek.")
        elif qv20 is not None:
            st.success(f"🚪 Likidite kapısı geçti (20g ort. hacim "
                       f"${qv20/1e6:.0f}M).")
        st.warning("⚠️ Alt işlemi büyük ölçüde KALDIRAÇLI BİR BTC "
                   "BAHSİ + coin'e özgü risktir. Sektörde tokenların "
                   "~%53'ü ölü (CoinGecko 2025) ve alt backtest'leri "
                   "hayatta-kalan yanlılığı taşır — boyutu küçük tut.")

    # ── REJİM ────────────────────────────────────────────────────────
    st.markdown(f"#### 🧭 {sembol} — rejim")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fiyat", f"{rejim['son']:,.4g}")
    m2.metric("200g MA", f"{rejim['ma200']:,.4g}",
              "ÜSTÜNDE" if rejim["trend"] else "ALTINDA",
              delta_color="off")
    m3.metric("12 hafta momentum",
              (f"%{rejim['mom12']*100:+.0f}"
               if rejim["mom12"] is not None else "—"), delta_color="off")
    m4.metric("30g vol (yıllık)",
              (f"%{rejim['vol30']*100:.0f}"
               if rejim["vol30"] is not None else "—"),
              (f"%{rejim['vol_pct']*100:.0f}. yüzdelik"
               if rejim["vol_pct"] is not None else None),
              delta_color="off")
    n1, n2, n3, n4 = st.columns(4)
    if rejim.get("mayer") is not None:
        n1.metric("Mayer (F/200g)", f"{rejim['mayer']:.2f}",
                  "aşırı >2.4 · dip <0.8", delta_color="off")
    if rejim.get("mvrv_z") is not None:
        n2.metric("MVRV-Z (BTC)", f"{rejim['mvrv_z']:.1f}",
                  "aşırı >6 · dip <0", delta_color="off")
    if fg:
        n3.metric("Fear&Greed", fg["deger"], fg.get("etiket"),
                  delta_color="off")
    if fund and fund.get("funding") is not None:
        n4.metric("Funding /8s", f"%{fund['funding']*100:.3f}",
                  delta_color="off")
    with st.container(border=True):
        st.markdown(f"**Boyut çarpanı: {rejim['carpan']:.2f}×** "
                    f"(0.70–1.30; hisse tarafıyla aynı kompozit mantık)")
        for g in rejim["gerekce"]:
            st.caption("• " + g)
        for b in rejim["bayraklar"]:
            st.caption(b + " — çarpana GİRMEZ, yalnız bayrak")

    # ── PLAN: AL → HEDEF → STOP ─────────────────────────────────────
    st.markdown("#### 🎯 Plan — al → net hedef → stop")
    pl = kripto_plan(df, rejim, k_atr=k_atr, alt_mi=alt_mi)
    if pl is None:
        st.error("Plan kurulamadı (veri yetersiz).")
        return
    if pl.get("gecersiz"):
        st.error("🛑 " + pl["not"])
        return
    if pl.get("tavan_notu"):
        st.caption("ℹ️ " + pl["tavan_notu"])
    boyut_pct = (risk / 100.0) / pl["stop_pct"]
    tutar = sermaye * boyut_pct
    asiri = boyut_pct > 0.25
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Giriş (spot)", f"{pl['giris']:,.6g}")
    p2.metric("Stop", f"{pl['stop']:,.6g}",
              f"−%{pl['stop_pct']*100:.1f} · {pl['kaynak']}",
              delta_color="off")
    p3.metric("ATR20", f"%{pl['atr_pct']*100:.1f}")
    p4.metric("Pozisyon", f"${min(tutar, sermaye*0.25):,.0f}",
              f"sermayenin %{min(boyut_pct,0.25)*100:.1f}"
              + (" (⇧ %25 tavan)" if asiri else ""),
              delta_color="off")
    st.caption(f"Boyut = risk({risk:.2f}%) ÷ stop({pl['stop_pct']*100:.1f}%) "
               f"× sermaye; rejim çarpanı {rejim['carpan']:.2f}× ile "
               f"nihai öneri ≈ **${min(tutar, sermaye*0.25)*rejim['carpan']:,.0f}**. "
               f"Alt için riski %0.25-0.5'e düşür.")
    tablo = []
    for s in pl["satirlar"]:
        g = s["gecis"]
        if g.get("yetersiz"):
            oran = f"veri az (N={g.get('n', 0)})"
            ga = "—"
        else:
            oran = f"%{g['oran']*100:.0f}"
            ga = (f"%{g['ga'][0]*100:.0f}–%{g['ga'][1]*100:.0f} "
                  f"(etkin N≈{g['etkin']})")
        tablo.append({
            "Hedef": f"{s['rr']:.1f}R",
            "Fiyat": f"{s['hedef']:,.6g}",
            "Getiri": f"+%{s['hedef_pct']*100:.1f}",
            "Net R:R (ücretli)": s["net_rr"],
            "Tarihsel ilk-geçiş": oran,
            "Wilson GA": ga})
    st.dataframe(pd.DataFrame(tablo), hide_index=True, width="stretch")
    st.caption("🗣️ **Sade okuma:** 'İlk-geçiş %46' = geçmişte bugünküne "
               "benzer trend/momentum günlerinden girilseydi, 90 gün "
               "içinde fiyat vakaların ~46'sında stoptan ÖNCE o hedefe "
               "ulaşmış demek. Gelecek garantisi DEĞİL.")
    st.warning("⚠️ " + pl["sinir"])
    with st.container(border=True):
        st.markdown("**NE DEMEK DEĞİL:**")
        st.markdown("- Bu bir fiyat tahmini DEĞİL — oranlar geçmiş "
                    "taban oranıdır; rejim değişince bozulur (ETF çağı).")
        st.markdown("- 'Kapı açık' = al demek DEĞİL; yalnız koşulların "
                    "tarihsel olarak aleyhte OLMADIĞINI söyler.")
        st.markdown("- Spot'ta tasfiye yok ama borsa/custody riski ve "
                    "fitil zararı VARDIR; stop-market kayabilir.")
    with st.expander("🚫 Bu ekranın KASITLI OLARAK kullanmadıkları"):
        for x in KRIPTO_REDDEDILENLER:
            st.markdown(f"- ✖ {x}")
        st.caption("CoinMetrics Community verisi CC BY-NC 4.0 — kişisel "
                   "kullanım içindir.")
    st.caption("Bu bir disiplin çerçevesidir; yatırım tavsiyesi değildir.")


def render_piyasa():
    """PİYASA BAĞLAMI ekranı v2 — Nasdaq / S&P rejimi + vade yapısı,
    kredi stresi, tranş planı ve hisse-endeks ilişkisi."""
    st.markdown("# 📉 Piyasa Bağlamı")
    st.error(
        "⚖️ **ÖNCE BUNU OKU.** Bu ekran sana *\"şurada al\"* demez; "
        "**endeks bugünküne benzer yerlerdeyken sonraki 3 ayın tarihsel "
        "olarak nasıl geçtiğini** gösterir — bir dağılım olarak. Endeks değerlemesinin 3-4 aylık ufukta öngörü gücü "
        "sıfıra yakındır — CAPE'in 1 *yıllık* ilişkisi bile pratikte "
        "sıfırdır (Invesco 1983-2024); anlamlı hale gelmesi ~10 yıl alır. "
        "Goyal-Welch (RFS 2008): öngörücülerin çoğu örneklem dışında "
        "başarısız. Bu ekranın söylediği şey: **hangi rejimdesin, o "
        "rejimde tarihsel dağılım nasıldı, ve pozisyonunu/temposunu ne "
        "kadar büyütüp küçültmelisin.**")

    c1, c2, c3 = st.columns([2, 1, 1])
    endeks = c1.selectbox("Endeks", list(ENDEKSLER),
                          format_func=lambda t: f"{ENDEKSLER[t]} ({t})")
    fk_el = c2.number_input(
        "Endeks F/K (opsiyonel)", 0.0, 60.0, 0.0, 0.5,
        help="Endeksin güncel F/K ya da CAPE'i. multpl.com veya "
             "Shiller verisinden alabilirsin. Boş bırakırsan değerleme "
             "bağlamı gösterilmez; girersen YAKLAŞIK Shiller ızgarasına göre "
             "konumlanır ve pozisyon çarpanına işler — UYDURULMAZ.")
    hisse_tk = c3.text_input(
        "Hisse (opsiyonel)", value="", key="pb_hisse",
        help="Almayı düşündüğün hisse (örn. CRCL, ISRG). Girersen "
             "hissenin endeksle beta / R² ilişkisi hesaplanır — bu "
             "ekranın SANA o hisse için ne kadar 'ek güven' "
             "verebileceğini gösterir (araştırma E).")

    if not st.button("📉 Bağlamı Getir", type="primary", width="stretch"):
        return

    with st.spinner("Endeks + vade yapısı + kredi verisi çekiliyor…"):
        kp = fetch_endeks(endeks)
        vx = fetch_endeks(VIX_TICKER, "10y")
        v3 = fetch_endeks(VIX3M_TICKER, "10y")
        oas = fetch_fred("BAMLH0A0HYM2")
        hyg = ief = None
        if oas is None:
            hyg = fetch_endeks("HYG", "10y")
            ief = fetch_endeks("IEF", "10y")
        tk = hisse_tk.strip().upper()
        hs = fetch_endeks(tk, "2y") if tk else None
        rf, _erp, _kay = rf_erp_guncel()
    if kp is None or len(kp) < 220:
        st.error("Endeks verisi alınamadı ya da yetersiz.")
        return

    vade = (vix_vade_yapisi(vx, v3)
            if (vx is not None and v3 is not None) else None)
    kredi = kredi_stresi(oas, hyg, ief)

    pk = piyasa_baglami(
        endeks, kp, vx, (float(vx.iloc[-1]) if vx is not None else None),
        None, (fk_el if fk_el > 0 else None), rf,
        vade=vade, kredi=kredi)
    if not pk:
        st.error("Bağlam hesaplanamadı.")
        return
    rj = pk["rejim"]

    # ── SADE ÖZET: bugünkü hava (tek metafor, ölçüme bağlı) ─────────
    hv = piyasa_hava_ozeti(pk)
    if hv:
        with st.container(border=True):
            st.markdown("### " + hv["baslik"])
            for _s in hv["satirlar"]:
                st.markdown("- " + _s)
            st.caption("⚖️ " + hv["not"])

    # ── REJİM ────────────────────────────────────────────────────────
    st.markdown(f"#### 🧭 {pk['endeks_adi']} — şu anki rejim")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seviye", f"{rj['son']:,.0f}")
    m2.metric("200G ortalama", f"{rj['ma200']:,.0f}",
              ("ÜSTÜNDE" if rj["ma200_ustu"] else "ALTINDA"),
              delta_color="off")
    m3.metric("52h zirveden", f"%{rj['cekilme']*100:.1f}",
              (rj.get("cekilme_kova") or {}).get("ad"), delta_color="off")
    if rj.get("vix") is not None:
        m4.metric("VIX", f"{rj['vix']:.1f}",
                  (f"%{rj['vix_yuzdelik']*100:.0f}. yüzdelik"
                   if rj.get("vix_yuzdelik") is not None else None),
                  delta_color="off")
    n1, n2, n3, n4 = st.columns(4)
    if vade:
        n1.metric("VIX / VIX3M", f"{vade['ivts']:.2f}",
                  ("HOOK ↩" if vade["hook"] else vade["durum"].upper()),
                  delta_color="off")
    if kredi:
        n2.metric("Kredi (HY)",
                  (f"%{kredi['deger']:.2f}"
                   if "OAS" in kredi["kaynak"] else f"{kredi['deger']:.3f}"),
                  f"z {kredi['z']:+.1f} · {kredi['durum']}",
                  delta_color="off")

    (st.success if rj["ma200_ustu"] else st.warning)(rj["trend_notu"])
    if rj.get("cekilme_kova"):
        st.info(f"**{rj['cekilme_kova']['ad'].upper()}** — "
                f"{rj['cekilme_kova']['aciklama']}")
    if endeks in ("^IXIC", "^NDX"):
        st.caption(
            "🧨 **Nasdaq notu (araştırma E):** Yukarıdaki kova metinleri "
            "S&P 500 tarihinden (T. Rowe Price 1940-2025) türetilmiştir "
            "ve Nasdaq için FAZLA İYİMSERDİR — Nasdaq 2000-02'de "
            "zirveden ~%78 düştü, nominal zirveye dönüşü ~15 yıl aldı. "
            "Aşağıdaki epizot sayımı ve taban oranları ise SEÇTİĞİN "
            "endeksin KENDİ verisinden hesaplanır; asıl onlara bak.")
    if rj.get("vix_kova"):
        st.info(f"**VIX: {rj['vix_kova']['ad'].upper()}** — "
                f"{rj['vix_kova']['aciklama']}")
    if vade:
        (st.warning if (vade["durum"] == "backwardation"
                        and not vade["hook"]) else st.info)(
            f"🌡️ **VADE YAPISI: "
            f"{'HOOK (geri dönüş)' if vade['hook'] else vade['durum'].upper()}"
            f"** — {vade['okuma']}")
    if kredi:
        (st.warning if kredi["z"] is not None and kredi["z"] >= 1
         else st.info)(f"🏦 **KREDİ [{kredi['kaynak']}]** — {kredi['okuma']}")

    # ── ÇEKİLME GEÇİŞİ — "buradan daha ne kadar düşer?" ─────────────
    st.markdown("#### 📉 \"Buradan daha ne kadar düşer?\"")
    st.caption("Ölçü: TÜM DÖNEM zirvesinden çekilme — yukarıdaki "
               "'52h zirveden' satırından FARKLI bir referanstır (O7).")
    ep = cekilme_epizotlari(kp)
    bn = bugun_nerede(kp, ep)
    dt = duz_turkce(bn, pk["endeks_adi"])
    if not dt:
        st.warning("Çekilme geçmişi hesaplanamadı (veri yetersiz).")
    elif not dt["satirlar"]:
        st.info("📊 " + dt["govde"])
    else:
        st.markdown(f"**{dt['basli']}**")
        st.markdown(dt["govde"])
        # tablo — DÜZ CÜMLE + fiyat karşılığı
        st.dataframe(pd.DataFrame([
            {"Buraya inerse": f"−%{s['hedef_yuzde']*100:.0f}",
             f"{pk['endeks_adi']} seviyesi": f"{s['hedef_fiyat']:,.0f}",
             "Tarihte kaç olayda": s["cumle"].replace("**", ""),
             "Güven aralığı": s["ga_cumle"]}
            for s in dt["satirlar"]]),
            hide_index=True, width="stretch")
        if dt["uyari"]:
            (st.error if "GÜVENME" in dt["uyari"] else st.info)(dt["uyari"])
        with st.expander("📊 Tüm çekilme seviyeleri için geçiş tablosu"):
            # gecis_tablosu() yazılmıştı ama HİÇ ÇAĞRILMIYORDU.
            # bugun_nerede() yalnız BUGÜNKÜ seviyeyi veriyor; bu tablo
            # her seviyeden her seviyeye geçişi gösterir.
            _gt = gecis_tablosu(ep)
            if _gt:
                _hedefler = [0.10, 0.15, 0.20, 0.25, 0.30]
                _sat = []
                for _s in _gt:
                    _r = {"Şu seviyeye indiyse": f"−%{_s['kosul']*100:.0f}",
                          "Kaç olay": _s["n_epizot"]}
                    for _h in _hedefler:
                        _g = _s["gecisler"].get(_h)
                        _r[f"→−%{_h*100:.0f}"] = (
                            f"%{_g['oran']*100:.0f}" if _g else "—")
                    _r["Yeterli mi"] = "evet" if _s["yeterli"] else "⚠️ N<10"
                    _sat.append(_r)
                st.dataframe(pd.DataFrame(_sat), hide_index=True,
                             width="stretch")
                st.caption(
                    "Satır: endeks şu derinliğe indiyse. Sütun: oradan "
                    "daha derine inme oranı. 'N<10' işaretli satırlara "
                    "GÜVENME — o kadar az olaydan oran çıkmaz.")

        with st.expander("📐 Bu sayılar nasıl hesaplandı? (oku — önemli)"):
            st.markdown(dt["tanim_notu"])
            st.caption(
                f"Toplam {bn['toplam_epizot']} olay tarandı; bugünkü "
                f"seviyeye ulaşan {bn['n_epizot']} tanesi kullanıldı. "
                f"Güven aralıkları Wilson yöntemiyle — küçük örneklemde "
                f"normal yaklaşım yanlış sonuç verir.")
            st.caption(
                "⚠️ En önemli sınır: bu oranlar birkaç tarihsel olaydan "
                "geliyor ve o olayların sebepleri BAMBAŞKA (2000 "
                "teknoloji balonu, 2008 kredi krizi, 2020 salgın, 2022 "
                "faiz). Onları aynı torbaya koymak, birbirine benzedikleri "
                "varsayımıdır — ve bu varsayım yanlış olabilir.")

    # ── TABAN ORANI (ileri getiri dağılımı) ─────────────────────────
    st.markdown("#### 🎯 \"Burası almak için iyi bir yer mi?\"")
    tb = taban_oranlari(kp)
    if tb.get("yetersiz"):
        st.warning("📊 " + tb.get("not", "Taban oranı hesaplanamadı."))
    else:
        gv = guvenli_mi(tb)
        _sade_tb = taban_orani_sade(tb)
        if _sade_tb:
            st.markdown(_sade_tb)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("3 ay medyan", f"%{tb['medyan']*100:+.1f}",
                  f"koşulsuz %{tb['kosulsuz_medyan']*100:+.1f}",
                  delta_color="off")
        t2.metric("Pozitif oran", f"%{tb['pozitif_oran']*100:.0f}",
                  f"koşulsuz %{tb['kosulsuz_pozitif']*100:.0f}",
                  delta_color="off")
        t3.metric("Kötü %10", f"%{tb['p10']*100:.1f}",
                  "en kötü onda bir", delta_color="off")
        t4.metric("%10+ kayıp", f"%{tb['kotu_oran']*100:.0f}",
                  "vakaların", delta_color="off")
        st.caption(f"Aralık: p25 %{tb['p25']*100:+.1f} … "
                   f"p75 %{tb['p75']*100:+.1f}  ·  "
                   f"{tb['gozlem']} gözlem "
                   f"(bağımsız ~{tb['etkin_gozlem']})")
        st.info("📊 " + gv["ozet"])
        st.warning("⚠️ " + gv["kotu_senaryo"])
        (st.success if "lehe" in gv["hukum"] else st.info)(
            "**HÜKÜM:** " + gv["hukum"])
        with st.expander("Bu sayılar ne kadar güvenilir?"):
            st.caption(tb["not"])
            st.caption(tb["sinir"])

    # ── POZİSYON ÇARPANI v2 ─────────────────────────────────────────
    pz = pk.get("pozisyon")
    if pz:
        st.markdown("#### ⚖️ Pozisyon Boyutu Çarpanı (v2 — kompozit skor)")
        st.caption("Yukarıdaki dağılım 'ne zaman' sorusuna kesin cevap "
                   "vermiyorsa, elinde kalan kol POZİSYON BOYUTU ve "
                   "TEMPODUR. v2: alt-sinyaller [-1,+1] skorlanır ve "
                   "ORTALANIR — çarpılmaz; aralık [0.70, 1.30].")
        pc1, pc2 = st.columns([1, 3])
        pc1.metric("Çarpan", f"{pz['carpan']:.2f}×",
                   f"ort. skor {pz['ort_skor']:+.2f}", delta_color="off")
        with pc2:
            for gk in pz["gerekce"]:
                st.caption("• " + gk)
        st.info("ℹ️ " + pz["not"])

    # ── TRANŞ PLANI — "şimdi mi, düşerse mi?" ───────────────────────
    st.markdown("#### 🪜 Tranş Planı — \"şimdi mi, düşerse mi?\"")
    tp = trans_plani(kp, (pz or {}).get("carpan", 1.0))
    if tp.get("yetersiz"):
        st.caption("🪜 " + tp.get("not", "Tranş planı hesaplanamadı."))
    else:
        pay = tp["pay"]
        st.markdown(
            f"Bugünkü rejim çarpanına ({(pz or {}).get('carpan', 1.0):.2f}×) "
            f"göre makul bölüşüm: **%{pay[0]} ŞİMDİ**, "
            f"%{pay[1]} birinci eşikte, %{pay[2]} ikinci eşikte.")
        _tr = [{"Tranş": f"1 — %{pay[0]}", "Tetik": "ŞİMDİ",
                f"{pk['endeks_adi']} seviyesi": f"{rj['son']:,.0f}",
                "Tarihsel dolum": "—",
                f"Bekleme maliyeti ({tp['ufuk_gun']}g medyan)": "—"}]
        for _i, _s in enumerate(tp["satirlar"]):
            _tr.append({
                "Tranş": f"{_i+2} — %{pay[_i+1]}",
                "Tetik": f"bugünden −%{_s['esik']*100:.0f}",
                f"{pk['endeks_adi']} seviyesi": f"{_s['seviye']:,.0f}",
                "Tarihsel dolum": f"%{_s['dolum_orani']*100:.0f}",
                f"Bekleme maliyeti ({tp['ufuk_gun']}g medyan)": (
                    f"%{_s['firsat_medyan']*100:+.1f}"
                    if _s["firsat_medyan"] is not None else "—")})
        st.dataframe(pd.DataFrame(_tr), hide_index=True, width="stretch")
        st.caption("🪜 " + tp["not"])
        st.caption(tp["sinir"])

    # ── HİSSE ↔ ENDEKS — 'ek güven' bu hisse için ne kadar geçerli? ─
    if tk:
        st.markdown(f"#### 🔗 {tk} ↔ {pk['endeks_adi']} — "
                    f"'ek güven' ne kadar geçerli?")
        if hs is None:
            st.warning(f"🔗 {tk} verisi alınamadı — sembolü kontrol et.")
        else:
            hi = hisse_endeks_iliskisi(hs, kp)
            if hi.get("yetersiz"):
                st.warning("🔗 " + hi.get("not", "Hesaplanamadı."))
            else:
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Beta", f"{hi['beta']:.2f}")
                h2.metric("Korelasyon", f"{hi['korelasyon']:.2f}")
                h3.metric("R²", f"%{hi['r2']*100:.0f}", hi["kova"],
                          delta_color="off")
                h4.metric("Ortak gün", f"{hi['gun']}")
                (st.success if hi["kova"] == "güçlü"
                 else st.info if hi["kova"] == "kısmi"
                 else st.warning)("🔗 " + hi["yorum"])
                if hi.get("ipo_uyari"):
                    st.error(hi["ipo_not"])
                st.caption(
                    "R² = bu hissenin günlük getiri oynaklığının endeksle "
                    "AÇIKLANAN payı (son ~2 yıl). Kalanı şirkete özgüdür "
                    "— endeks ekranı o kısmı GÖREMEZ.")

    # ── DEĞERLEME ───────────────────────────────────────────────────
    if pk.get("deger"):
        d = pk["deger"]
        st.markdown("#### 📊 Değerleme konumu")
        st.metric(d["olcut"], f"{d['deger']:.1f}",
                  f"%{d['yuzdelik']*100:.0f}. yüzdelik", delta_color="off")
        st.warning(d["ufuk_uyarisi"])
    else:
        st.caption("📊 Değerleme bağlamı için yukarıdaki alana endeks "
                   "F/K veya CAPE gir — girilen değer YAKLAŞIK Shiller ızgarasına "
                   "göre konumlanır; girilmezse UYDURULMAZ.")

    # ── YOĞUNLAŞMA + FAİZ ───────────────────────────────────────────
    if pk.get("yogunlasma"):
        st.markdown("#### 🎯 Yoğunlaşma")
        st.warning(pk["yogunlasma"]["uyari"])
    if pk.get("faiz"):
        st.markdown("#### 🏦 Faiz bağlamı")
        st.caption(f"10Y: %{(rf or 0)*100:.2f} ({_kay})")
        st.info(pk["faiz"]["not"])

    # ── REDDEDİLENLER ───────────────────────────────────────────────
    with st.expander("🚫 Bu ekranın KASITLI OLARAK yapmadıkları"):
        st.caption("Aşağıdakiler kanıt yetersizliği nedeniyle bilerek "
                   "eklenmedi. Liste koda yazılı ki ileride yeniden "
                   "eklenmesin.")
        for x in pk["reddedilenler"]:
            st.markdown(f"- ✖ {x}")

    st.caption("Bu bir disiplin çerçevesidir; yatırım tavsiyesi değildir.")


def render_rakip():
    st.markdown("# ⚔️ Rakip Kıyası")
    st.caption("Belirlediğin 2-8 hisseyi yan yana kıyaslar: çarpanlar, "
               "kârlılık, büyüme, analist görünümü, trend/momentum ve "
               "hızlı skorlar — her metrikte en iyiye 🏅 rozet, medyan "
               "sütunu ve 4 eksenli GRUP İÇİ sıralamayla. Hafif veriyle "
               "çalışır; nihai hüküm için adayı Derin Analize gönder "
               "(ABD hisseleri).")
    ham = st.text_input("Hisseler (virgül/boşlukla, 2-8)",
                        value="", key="rk_giris",
                        help="Ör: NVDA, AMD, INTC")
    tickers = tuple(dict.fromkeys(
        t.strip().upper() for t in ham.replace(",", " ").split()
        if t.strip()))[:8]
    if st.button("⚔️ Kıyasla", type="primary", width="stretch",
                 disabled=len(tickers) < 2):
        veriler, yok, kapanislar = [], [], {}
        prog = st.progress(0.0, text="Veri çekiliyor…")
        for i, t in enumerate(tickers):
            d = hizli_veri(t)
            prog.progress((i + 1) / len(tickers), text=t)
            if d is None:
                yok.append(t)
                continue
            # KIYAS v2 için fiyat geçmişi de gerekiyor (12-1 momentum)
            try:
                kapanislar[t] = fetch_kapanis_seri(t, "2y")
            except Exception:
                kapanislar[t] = None
            veriler.append({"t": t, "isim": d["isim"],
                            "cur": ("₺" if d.get("currency") == "TRY"
                                    else "$"),
                            "sk": hizli_skorlar(d),
                            "_ham": d})
        prog.empty()
        st.session_state["rakip"] = {"veriler": veriler, "yok": yok,
                                     "kapanislar": kapanislar}

    rk = st.session_state.get("rakip")
    if not rk:
        return
    veriler, yok = rk["veriler"], rk["yok"]
    if yok:
        st.caption(f"Veri alınamayan semboller (atlandı): {', '.join(yok)}")

    # ── İKİ SEKME: karar motoru ÖNCE, ham metrik tablosu SONRA ───────
    _s1, _s2 = st.tabs(["🎯 Karar (3-4 ay ufku)", "📊 Ham Metrik Tablosu"])
    with _s1:
        try:
            render_kiyas_v2(veriler, rk.get("kapanislar") or {})
        except Exception as _e:
            st.error(f"Kıyas motoru hata verdi: {type(_e).__name__}")
            with st.expander("Teknik ayrıntı"):
                st.code(traceback.format_exc())
    with _s2:
        st.caption("Aşağıdaki tablo TÜM metrikleri gösterir — değer ve "
                   "kalite dahil. Bunlar 3-4 aylık sıralamaya girmez "
                   "(o ufukta ayrıştırmıyorlar) ama şirketi ANLAMAK için "
                   "değerlidir ve uzun vadeli bakışta esastır.")
        _render_rakip_tablo(veriler)


def _render_rakip_tablo(veriler):
    """Eski metrik tablosu — kaldırılmadı, ikinci sekmeye alındı."""
    oz = rakip_ozet(veriler)
    if not oz:
        st.warning("Kıyas için en az 2 sembolden veri gerekli.")
        return
    sira = oz["sira"]
    curmap = {v["t"]: v["cur"] for v in veriler}
    isimmap = {v["t"]: v["isim"] for v in veriler}
    st.caption(" · ".join(f"**{t}** {isimmap[t]}" for t in sira))

    pct = ("Brüt Marj", "Faaliyet Marjı", "Net Marj", "ROE",
           "Gelir Büyümesi", "52h Konum", "FCF Marjı",
           "Açığa Satış %Float")
    carpan = ("F/K (fwd)", "F/K (trl)", "EV/EBITDA", "F/S", "P/FCF",
              "PEG", "P/B", "NB/EBITDA", "Cari Oran")

    def fmt_hucre(ad, t, x):
        if x is None:
            return "—"
        if ad == "Fiyat":
            return f"{curmap[t]}{x:,.2f}"
        if ad == "Piyasa Değeri":
            return fmt_money(x, curmap[t])
        if ad in ("Analist Potansiyeli", "200G Trend", "52h Değişim"):
            return f"%{x*100:+.0f}"
        if ad == "Analist Tavsiye (1-5)":
            return f"{x:.1f}"
        if ad == "Beta":
            return f"{x:.2f}"
        if ad in pct:
            return f"%{x*100:.0f}"
        if ad in carpan:
            return f"{x:.1f}x"
        return f"{x:.0f}" if ad in ("Kalite", "Pahalılık",
                                    "Seçim Puanı") else f"{x}"

    rozset = {t: set(oz["rozetler"][t]) for t in sira}
    satirlar = []
    for ad, degerler in oz["metrikler"].items():
        row = {"Metrik": ad}
        for t in sira:
            hucre = fmt_hucre(ad, t, degerler.get(t))
            r_adi = ("en düşük " + ad
                     if ("en düşük " + ad) in rozset[t]
                     else "en yüksek " + ad
                     if ("en yüksek " + ad) in rozset[t] else None)
            row[t] = ("🏅 " + hucre) if r_adi else hucre
        md = oz["medyan"].get(ad)
        row["Medyan"] = (fmt_hucre(ad, sira[0], md)
                         if ad not in ("Fiyat", "Piyasa Değeri") and
                         md is not None else "—")
        satirlar.append(row)
    st.dataframe(pd.DataFrame(satirlar), hide_index=True,
                 width="stretch", height=min(740, 60 + 36 *
                                                      len(satirlar)))

    fig = go.Figure()
    # B11-4c: ÖLÇÜM — #3fb27f yeşil vs #e05c4b kırmızı, algısal fark
    # normal görüşte 320, dötanopide 98 (%69 çöküş): iki çubuk zeytin
    # rengine yakınsıyor (#9B9B81 vs #939343). Etiket olsa bile çubuk
    # grafiğin asıl işi olan ANINDA KARŞILAŞTIRMA kayboluyor.
    # Wong mavi-turuncu: 467 → 358 (korunuyor).
    fig.add_bar(name="Kalite", x=list(sira),
                y=[oz["metrikler"]["Kalite"][t] for t in sira],
                marker_color=WONG["gok"])
    fig.add_bar(name="Pahalılık", x=list(sira),
                y=[oz["metrikler"]["Pahalılık"][t] for t in sira],
                marker_color=WONG["turuncu"])
    fig.update_layout(barmode="group",
                      title="Kalite vs Pahalılık (0-100)", **_LAYOUT)
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    gs = oz.get("grup_sira") or []
    ep = oz.get("eksen_puan") or {}
    if gs:
        st.markdown("#### 🏁 Grup İçi Sıralama (yüzdelik konum)")
        st.dataframe(pd.DataFrame(
            [{"Sıra": i + 1, "Ticker": t,
              "Genel": oz["genel_puan"].get(t),
              "Değerleme": (ep.get(t) or {}).get("Değerleme"),
              "Kalite": (ep.get(t) or {}).get("Kalite"),
              "Büyüme": (ep.get(t) or {}).get("Büyüme"),
              "Momentum": (ep.get(t) or {}).get("Momentum")}
             for i, t in enumerate(gs)]),
            hide_index=True, width="stretch")
        for t in gs:
            g_ = (oz.get("gerekce") or {}).get(t)
            if g_:
                st.caption(f"**{t}:** {g_}")
        st.caption("⚠️ " + (oz.get("siralama_notu") or ""))

    st.markdown("**🏅 Rozet dağılımı:**")
    for t in sira:
        r_ = oz["rozetler"][t]
        st.caption(f"**{t}** ({len(r_)}): "
                   + (", ".join(r_) if r_ else "—"))
    st.caption(f"Rozet lideri: **{oz['rozet_lideri']}** — ama rozet "
               f"sayısı hüküm değildir; kalite-pahalılık dengesine ve "
               f"uyarılara birlikte bak.")
    for u in oz.get("uyarilar") or []:
        st.warning(u)
    st.caption("Skorlar hafif-veri hızlı skorlarıdır (ön kıyas); Beneish, "
               "SEC çaprazı, DCF ve katalizörler yalnız derin analizde. "
               "Yatırım tavsiyesi değildir.")

    rc1, rc2 = st.columns([2, 1])
    sec = rc1.selectbox("Derin analize gönderilecek hisse", list(sira),
                        key="rk_sec")
    if rc2.button("🔬 Derin Analize Gönder", width="stretch",
                  key="rk_git"):
        st.session_state["_git_tkr"] = sec
        st.session_state["_git_mod"] = MOD_TEK
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# UI ADAPTÖRÜ — özel HTML kart sistemi (yapısal yeniden tasarım)
# st.metric/st.info duvarları yerine: bento KPI, hüküm banner'ı,
# conic-gradient gauge, hero. YALNIZ görsel; hesap/mantık değişmez.
# Kullanıcı verisi daima esc() ile kaçırılır (XSS koruması).
# ═══════════════════════════════════════════════════════════════════
import html as _html_mod


def esc(x):
    return _html_mod.escape(str(x))


def _ui_tok():
    if _tema_koyu_mu():
        return dict(bg="#0b0e14",
                    card="#131722", line="rgba(255,255,255,0.09)",
                    txt="#e7ecf3", mut="rgba(231,236,243,0.60)",
                    accent="#6366f1", ok="#22c55e", warn="#f59e0b",
                    bad="#ef4444", info="#38bdf8")
    return dict(bg="#fafaf9",
                card="#ffffff", line="rgba(0,0,0,0.09)",
                txt="#1c1c1e", mut="rgba(28,28,30,0.60)",
                accent="#4f46e5", ok="#16a34a", warn="#d97706",
                bad="#dc2626", info="#0284c7")


def _hero(baslik, alt="", ust=""):
    ustp = f'<div class="hero-eyebrow">{esc(ust)}</div>' if ust else ""
    altp = f"<p>{esc(alt)}</p>" if alt else ""
    st.markdown(f'<div class="hero">{ustp}<h1>{esc(baslik)}</h1>{altp}'
                f'<div class="hero-cizgi"></div></div>',
                unsafe_allow_html=True)


def _sparkline_svg(vals, renk, w=132, h=36):
    """Kapanış serisinden mini SVG çizgi (kart içi canlı grafik)."""
    try:
        v = [float(x) for x in vals if x == x][-60:]
        if len(v) < 8:
            return ""
        lo, hi = min(v), max(v)
        rng = (hi - lo) or 1.0
        n = len(v)
        pts = []
        for i, x in enumerate(v):
            px = i * (w - 4) / (n - 1) + 2
            py = h - 3 - (x - lo) / rng * (h - 9)
            pts.append(f"{px:.1f},{py:.1f}")
        yol = " ".join(pts)
        sx, sy = pts[-1].split(",")
        return (f'<svg class="spark" width="{w}" height="{h}" '
                f'viewBox="0 0 {w} {h}">'
                f'<polygon points="2,{h-2} {yol} {w-2},{h-2}" '
                f'fill="{renk}20"/>'
                f'<polyline points="{yol}" fill="none" stroke="{renk}" '
                f'stroke-width="2" stroke-linecap="round"/>'
                f'<circle cx="{sx}" cy="{sy}" r="2.6" fill="{renk}"/>'
                f'</svg>')
    except Exception:
        return ""


def kpi_serit(kartlar):
    """kartlar: [(etiket, değer, alt|None, ton[, ekstra_html]), ...]
    5. eleman opsiyonel: kartın içine gömülecek HTML (örn. sparkline)."""
    T = _ui_tok()
    h = []
    for kart in kartlar:
        lab, val, sub, ton = kart[:4]
        ek = kart[4] if len(kart) > 4 and kart[4] else ""
        renk = T.get(ton or "accent", T["accent"])
        altp = (f'<div style="color:{renk};font-size:.76rem;margin-top:3px;'
                f'font-weight:600">{esc(sub)}</div>') if sub else ""
        h.append(f'<div class="kpi" style="--ka:{renk}">'
                 f'<div class="kpi-l">{esc(lab)}</div>'
                 f'<div class="kpi-v">{esc(val)}</div>{altp}{ek}</div>')
    st.markdown('<div class="kpi-grid">' + "".join(h) + "</div>",
                unsafe_allow_html=True)


def rozet(text, ton="accent"):
    T = _ui_tok()
    renk = T.get(ton, T["accent"])
    return (f'<span style="background:{renk}22;color:{renk};'
            f'border:1px solid {renk}55;border-radius:999px;'
            f'padding:2px 10px;font-size:.75rem;font-weight:600;'
            f'margin-right:6px">{esc(text)}</span>')


def gauge(pct, label, ton="ok"):
    """0-100 → süpürerek dolan halka gösterge (CSS @property; JS yok)."""
    T = _ui_tok()
    renk = T.get(ton, T["ok"])
    p = max(0, min(100, int(round(pct or 0))))
    st.markdown(
        f'<div class="gauge-kap">'
        f'<div class="gauge-ring" style="--gp:{p*3.6:.0f}deg;--rk:{renk}">'
        f'<div><span class="gauge-say">{p}</span>'
        f'<span class="gauge-b">/100</span></div></div>'
        f'<span class="gauge-l">{esc(label)}</span></div>',
        unsafe_allow_html=True)


def verdict_banner(label, emoji, kind, msg):
    T = _ui_tok()
    renk = {"success": T["ok"], "warning": T["warn"],
            "error": T["bad"], "info": T["info"]}.get(kind, T["info"])
    st.markdown(
        f'<div class="vb" style="--vk:{renk}">'
        f'<div class="vb-e">{emoji}</div>'
        f'<div><div class="vb-l">HÜKÜM: {esc(label)}</div>'
        f'<div class="vb-m">{esc(msg)}</div></div></div>',
        unsafe_allow_html=True)





# ═══════════════════════════════════════════════════════════════════
# JS SAHNE KATMANI — iframe içinde GERÇEK JavaScript (vitrin şartnamesi)
# components.html iframe'i: JS çalışır, CDN yüklenir (Community Cloud
# CSP koymaz — issue #9160), Python→JS köprüsü _json_guvenli ile XSS
# emniyetli. Her sahne_* False dönerse çağıran CSS kartlara düşer.
# Sert sınırlar: iframe her rerun'da yeniden yüklenir (animasyon
# tekrar oynar), yükseklik sabittir, içinden Python'a tık gitmez.
# ═══════════════════════════════════════════════════════════════════
_SAHNE_BAS = """<!DOCTYPE html><html lang="tr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:%(bg)s;--card:%(card)s;--line:%(line)s;--accent:%(accent)s;
--ok:%(ok)s;--warn:%(warn)s;--bad:%(bad)s;--txt:%(txt)s;--mut:%(mut)s}
*{margin:0;padding:0;box-sizing:border-box}
html,body{background:%(bg)s;color:var(--txt);height:100%%;
font-family:'Inter',system-ui,sans-serif;-webkit-font-smoothing:antialiased;
overflow:hidden}
h1,h2,h3,.disp{font-family:'Space Grotesk','Inter',sans-serif}
.tnum{font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
transition:none!important}}
</style></head><body>
<script>const DATA=%(data)s;const TOK=%(tok)s;</script>
%(govde)s
</body></html>"""


def _json_guvenli(obj):
    """<script> içine gömülmeye hazır JSON: </script> kaçışı, <,>,&
    ve U+2028/29 nötrlenir (XSS-güvenli köprü)."""
    s = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)
    return (s.replace("<", "\\u003c").replace(">", "\\u003e")
             .replace("&", "\\u0026")
             .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _sahne_belge(govde, veri):
    T = _ui_tok()
    return _SAHNE_BAS % {
        "bg": T["bg"], "card": T["card"], "line": T["line"],
        "accent": T["accent"], "ok": T["ok"], "warn": T["warn"],
        "bad": T["bad"], "txt": T["txt"], "mut": T["mut"],
        "data": _json_guvenli(veri), "tok": _json_guvenli(T),
        "govde": govde}


def _sahne(govde, veri, yukseklik):
    """Belgeyi kur, sürüm-bağımsız iframe'e bas. Hata → False
    (çağıran CSS fallback'ine düşer; sayfa asla boş kalmaz)."""
    try:
        belge = _sahne_belge(govde, veri)
        if hasattr(st, "iframe"):          # >=1.56 yeni API
            try:
                st.iframe(belge, height=yukseklik)
                return True
            except Exception:
                pass
        import streamlit.components.v1 as _components
        _components.html(belge, height=yukseklik, scrolling=False)
        return True
    except Exception:
        return False


# ── ŞABLON 1: HERO + KPI BOARD (392px) ─────────────────────────────
_HERO_JS = r"""
<canvas id="orb"></canvas>
<div class="wrap">
  <div class="eyebrow" id="eb"></div>
  <h1 class="title" id="ti"></h1>
  <div class="sub" id="su"></div>
  <div class="kpis" id="kp"></div>
  <canvas id="spark"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/countup.js/2.8.0/countUp.umd.js"></script>
<style>
.wrap{position:relative;z-index:2;padding:20px 24px 14px;height:100%;
  display:flex;flex-direction:column}
#orb{position:absolute;inset:0;width:100%;height:100%;z-index:1;opacity:.5}
.eyebrow{letter-spacing:.2em;font-size:11px;color:var(--accent);
  text-transform:uppercase;font-weight:700}
.title{font-size:30px;font-weight:700;margin:3px 0 1px;line-height:1.05;
  background:linear-gradient(94deg,var(--txt),var(--accent) 130%);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--mut);font-size:12.5px;margin-bottom:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  gap:11px}
.kpi{background:color-mix(in srgb,var(--card) 78%,transparent);
  border:1px solid var(--line);border-radius:15px;padding:12px 14px;
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  opacity:0;transform:translateY(10px);animation:rise .55s forwards}
.kpi .lab{font-size:10px;color:var(--mut);text-transform:uppercase;
  letter-spacing:.09em;font-weight:700}
.kpi .val{font-size:21px;font-weight:650;margin-top:4px;color:var(--txt)}
.kpi:first-child .val{font-size:25px;
  background:linear-gradient(100deg,var(--txt),var(--accent));
  -webkit-background-clip:text;background-clip:text;color:transparent}
@keyframes rise{to{opacity:1;transform:none}}
#spark{width:100%;height:56px;margin-top:12px}
</style>
<script>
(function(){
  var esc=function(s){return String(s==null?"":s);};
  document.getElementById('eb').textContent=esc(DATA.ticker)+
    (DATA.sector?' · '+esc(DATA.sector):'');
  document.getElementById('ti').textContent=esc(DATA.name);
  document.getElementById('su').textContent=[esc(DATA.industry),
    esc(DATA.mcap_txt)].filter(Boolean).join(' · ');

  var kpis=[
    {lab:'Fiyat',end:DATA.price,pre:DATA.cur,dec:2},
    {lab:DATA.fair_lbl||'Değer Bandı',txt:DATA.fair_txt||'—'},
    {lab:'Güvenlik Marjı',end:DATA.mos,suf:'%',dec:1},
    {lab:'Kalite',end:DATA.quality,suf:'/100',dec:0},
    {lab:'Piyasa Değeri',txt:DATA.mcap_txt||'—'}
  ];
  var box=document.getElementById('kp');
  kpis.forEach(function(k,i){
    var d=document.createElement('div');d.className='kpi';
    d.style.animationDelay=(i*80)+'ms';
    var lab=document.createElement('div');lab.className='lab';
    lab.textContent=k.lab;
    var val=document.createElement('div');val.className='val tnum';
    val.id='v'+i;d.appendChild(lab);d.appendChild(val);box.appendChild(d);
    if(k.txt!==undefined){val.textContent=k.txt;return;}
    if(k.end==null){val.textContent='—';return;}
    if(window.countUp&&countUp.CountUp){
      new countUp.CountUp('v'+i,k.end,{duration:1.4,
        decimalPlaces:k.dec||0,prefix:k.pre||'',suffix:k.suf||'',
        separator:',',decimal:'.'}).start();
    }else{val.textContent=(k.pre||'')+k.end.toFixed(k.dec||0)+(k.suf||'');}
  });

  var c=document.getElementById('spark');
  var a=DATA.closes||[];
  if(a.length>7){
    var x=c.getContext('2d');
    var w=c.width=c.offsetWidth*2,h=c.height=112;
    var mn=Math.min.apply(null,a),mx=Math.max.apply(null,a),n=a.length;
    var px=function(i){return i/(n-1)*w;};
    var py=function(v){return h-((v-mn)/((mx-mn)||1))*(h-18)-9;};
    var g=x.createLinearGradient(0,0,0,h);
    g.addColorStop(0,TOK.accent+'59');g.addColorStop(1,TOK.accent+'00');
    x.beginPath();x.moveTo(0,h);
    a.forEach(function(v,i){x.lineTo(px(i),py(v));});
    x.lineTo(w,h);x.closePath();x.fillStyle=g;x.fill();
    x.beginPath();
    a.forEach(function(v,i){i?x.lineTo(px(i),py(v)):x.moveTo(px(i),py(v));});
    x.strokeStyle=(a[n-1]>=a[0])?TOK.ok:TOK.bad;
    x.lineWidth=3;x.lineJoin='round';x.stroke();
  }else{c.style.display='none';}

  var oc=document.getElementById('orb'),o=oc.getContext('2d');
  oc.width=oc.offsetWidth;oc.height=oc.offsetHeight;
  var orbs=[0,1,2,3,4].map(function(){return{
    x:Math.random()*oc.width,y:Math.random()*oc.height,
    r:55+Math.random()*85,dx:(Math.random()-.5)*.3,
    dy:(Math.random()-.5)*.3};});
  var dur=matchMedia('(prefers-reduced-motion: reduce)').matches;
  function ciz(){o.clearRect(0,0,oc.width,oc.height);
    orbs.forEach(function(b){
      var g=o.createRadialGradient(b.x,b.y,0,b.x,b.y,b.r);
      g.addColorStop(0,TOK.accent+'3d');g.addColorStop(1,TOK.accent+'00');
      o.fillStyle=g;o.beginPath();o.arc(b.x,b.y,b.r,0,7);o.fill();
      b.x+=b.dx;b.y+=b.dy;
      if(b.x<0||b.x>oc.width)b.dx*=-1;
      if(b.y<0||b.y>oc.height)b.dy*=-1;});
    if(!dur)requestAnimationFrame(ciz);}
  ciz();
})();
</script>
"""

# ── ŞABLON 2: HÜKÜM SAHNESİ — ECharts glow gauge (344px) ───────────
_HUKUM_JS = r"""
<div class="stage">
  <div id="g1" class="gauge"></div>
  <div class="center">
    <div class="squircle" id="emo"></div>
    <div class="banner disp" id="lab"></div>
    <div class="msg" id="msg"></div>
  </div>
  <div id="g2" class="gauge"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
.stage{display:grid;grid-template-columns:1fr 1.15fr 1fr;
  align-items:center;height:100%;padding:6px 4px}
.gauge{height:300px;min-width:0}
.center{text-align:center;padding:0 10px}
.squircle{width:62px;height:62px;margin:0 auto 10px;border-radius:21px;
  display:grid;place-items:center;font-size:32px;
  background:color-mix(in srgb,var(--txt) 6%,transparent);
  border:1px solid var(--line)}
.banner{font-size:22px;font-weight:700;margin-bottom:6px;
  letter-spacing:-.01em}
.msg{font-size:12.5px;color:var(--mut);line-height:1.55;max-width:46ch;
  margin:0 auto}
.fallb{display:grid;place-items:center;height:100%;color:var(--txt)}
.fallb b{font-size:40px;display:block}
</style>
<script>
(function(){
  var KIND={success:TOK.ok,warning:TOK.warn,error:TOK.bad,info:TOK.accent};
  var col=KIND[DATA.kind]||TOK.accent;
  var emo=document.getElementById('emo');
  emo.textContent=DATA.emoji||'⚖️';
  emo.style.boxShadow='0 0 42px '+col+'55';
  var lab=document.getElementById('lab');
  lab.textContent='HÜKÜM: '+(DATA.label||'');
  lab.style.color=col;lab.style.textShadow='0 0 26px '+col+'66';
  document.getElementById('msg').textContent=DATA.msg||'';

  function qcol(v){return v>=60?TOK.ok:v>=40?TOK.warn:TOK.bad;}
  function pcol(v){return v<35?TOK.ok:v<=55?TOK.warn:TOK.bad;}

  function bas(id,val,ad,c){
    var el=document.getElementById(id);
    if(val==null){el.innerHTML='<div class="fallb"><div>'
      +'<b>—</b>'+ad+'</div></div>';return;}
    if(typeof echarts==='undefined'){
      el.innerHTML='<div class="fallb"><div style="color:'+c+'">'
        +'<b class="tnum">'+Math.round(val)+'</b>'+ad+'</div></div>';
      return;}
    var ch=echarts.init(el,null,{renderer:'canvas'});
    ch.setOption({series:[{type:'gauge',center:['50%','56%'],
      radius:'86%',startAngle:225,endAngle:-45,min:0,max:100,
      animationDuration:1500,animationEasing:'cubicOut',
      axisLine:{roundCap:true,lineStyle:{width:15,
        color:[[1,'rgba(148,163,184,0.14)']]}},
      progress:{show:true,roundCap:true,width:15,itemStyle:{
        color:{type:'linear',x:0,y:1,x2:1,y2:0,colorStops:[
          {offset:0,color:c},{offset:1,color:TOK.accent}]},
        shadowColor:c+'99',shadowBlur:20}},
      pointer:{show:true,length:'58%',width:4,
        itemStyle:{color:TOK.txt,shadowColor:c+'aa',shadowBlur:10}},
      anchor:{show:true,showAbove:true,size:14,itemStyle:{
        color:TOK.bg,borderColor:c,borderWidth:4,
        shadowColor:c+'cc',shadowBlur:10}},
      axisTick:{show:false},splitLine:{show:false},axisLabel:{show:false},
      detail:{valueAnimation:true,formatter:'{value}',fontSize:40,
        fontWeight:'bolder',color:TOK.txt,offsetCenter:[0,'34%'],
        fontFamily:'Space Grotesk'},
      title:{show:true,offsetCenter:[0,'64%'],fontSize:14,color:TOK.mut},
      data:[{value:Math.round(val),name:ad}]}]});
    addEventListener('resize',function(){ch.resize();});
  }
  bas('g1',DATA.quality,'Kalite',
      DATA.quality==null?TOK.mut:qcol(DATA.quality));
  bas('g2',DATA.score,'Pahalılık',
      DATA.score==null?TOK.mut:pcol(DATA.score));
})();
</script>
"""

# ── ŞABLON 3: FİYAT PANELİ — lightweight-charts v5 (466px) ─────────
_FIYAT_JS = r"""
<div class="leg" id="leg"></div>
<div id="chart"></div>
<script src="https://unpkg.com/lightweight-charts@5.0.8/dist/lightweight-charts.standalone.production.js"></script>
<style>
#chart{width:100%;height:404px}
.leg{display:flex;gap:14px;padding:8px 12px 4px;font-size:12px;
  color:var(--mut);align-items:baseline;flex-wrap:wrap}
.leg b{color:var(--txt)}
.leg .t{font-weight:700;color:var(--txt);margin-right:4px;
  font-family:'Space Grotesk'}
.hata{padding:44px;text-align:center;color:var(--mut)}
</style>
<script>
(function(){
  var el=document.getElementById('chart');
  if(!window.LightweightCharts){
    el.innerHTML='<div class="hata">Grafik motoru yüklenemedi (CDN) — '
      +'sekmelerdeki grafikler geçerlidir.</div>';return;}
  var LWC=window.LightweightCharts;
  var chart=LWC.createChart(el,{
    layout:{background:{type:'solid',color:'transparent'},
      textColor:TOK.mut,fontFamily:'Inter'},
    grid:{vertLines:{color:'rgba(148,163,184,.07)'},
      horzLines:{color:'rgba(148,163,184,.07)'}},
    rightPriceScale:{borderColor:'rgba(148,163,184,.12)'},
    timeScale:{borderColor:'rgba(148,163,184,.12)'},
    crosshair:{mode:1,
      vertLine:{color:TOK.accent,labelBackgroundColor:TOK.accent},
      horzLine:{color:TOK.accent,labelBackgroundColor:TOK.accent}},
    autoSize:true});
  var s=chart.addSeries(LWC.CandlestickSeries,{
    upColor:TOK.ok,downColor:TOK.bad,borderUpColor:TOK.ok,
    borderDownColor:TOK.bad,wickUpColor:TOK.ok,wickDownColor:TOK.bad});
  s.setData(DATA.ohlc||[]);
  function pl(fiyat,renk,ad,stil){
    if(fiyat==null)return;
    s.createPriceLine({price:fiyat,color:renk,lineWidth:1,
      lineStyle:stil,axisLabelVisible:true,title:ad});}
  var b=DATA.bant||{};
  pl(b.p75,TOK.warn,' MC P75',2);
  pl(b.p25,TOK.ok,' MC P25',2);
  pl(b.fair,'#a78bfa',' adil',0);
  (DATA.dilimler||[]).forEach(function(d){
    pl(d.seviye,TOK.accent,' '+d.label,3);});
  pl(DATA.izleme,TOK.mut,' izleme',1);
  chart.timeScale().fitContent();
  var son=(DATA.ohlc||[]).slice(-1)[0];
  if(son){document.getElementById('leg').innerHTML=
    '<span class="t">'+String(DATA.cur||'')+'</span>'+
    '<span>A <b class="tnum">'+son.open+'</b></span>'+
    '<span>Y <b class="tnum">'+son.high+'</b></span>'+
    '<span>D <b class="tnum">'+son.low+'</b></span>'+
    '<span>K <b class="tnum">'+son.close+'</b></span>'+
    '<span style="margin-left:auto;color:'+TOK.mut+'">kesikli: MC bandı'
    +' · mavi: kademe dilimleri</span>';}
})();
</script>
"""


def sahne_hero(m, fair_lbl, fair_txt):
    """Şirket hero + KPI panosu (count-up + canvas). False → CSS'e düş."""
    try:
        closes = []
        h = m.get("_hist")
        if h is not None and "Close" in getattr(h, "columns", []):
            closes = [round(float(x), 4) for x in
                      pd.Series(h["Close"]).dropna().tail(60)]
        veri = {
            "name": m.get("name") or m.get("ticker") or "",
            "ticker": m.get("ticker") or "",
            "sector": m.get("sector") or "",
            "industry": m.get("industry") or "",
            "cur": m.get("cur") or "$",
            "price": float(m["price"]) if m.get("price") else None,
            "fair_lbl": fair_lbl, "fair_txt": fair_txt,
            "mos": (float(m["mos"]) * 100
                    if m.get("mos") is not None else None),
            "quality": (float(m["quality"])
                        if m.get("quality") is not None else None),
            "mcap_txt": fmt_money(m.get("mcap"), m.get("cur") or "$"),
            "closes": closes}
        return _sahne(_HERO_JS, veri, 392)
    except Exception:
        return False


def sahne_hukum(m):
    """Hüküm sahnesi: iki ışıyan gauge + merkez banner. False → CSS."""
    try:
        label, emoji, kind, msg = m["verdict"]
        veri = {"label": str(label), "emoji": str(emoji),
                "kind": str(kind), "msg": str(msg),
                "quality": (float(m["quality"])
                            if m.get("quality") is not None else None),
                "score": (float(m["score"])
                          if m.get("score") is not None else None)}
        return _sahne(_HUKUM_JS, veri, 344)
    except Exception:
        return False


def sahne_fiyat(m):
    """TradingView-motoru mum paneli: MC bandı + kademe dilimleri
    fiyat çizgisi olarak işlenir. Veri kısaysa sessizce atlanır."""
    try:
        h = m.get("_hist")
        if h is None or len(h) < 40:
            return False
        kolonlar = set(getattr(h, "columns", []))
        if not {"Open", "High", "Low", "Close"}.issubset(kolonlar):
            return False
        d = h.tail(500)
        idx = pd.to_datetime(d.index)
        gorulen, ohlc = set(), []
        for ts, o, y, lo, c in zip(idx, d["Open"], d["High"],
                                   d["Low"], d["Close"]):
            gun = ts.strftime("%Y-%m-%d")
            if gun in gorulen:
                continue
            gorulen.add(gun)
            ohlc.append({"time": gun, "open": round(float(o), 4),
                         "high": round(float(y), 4),
                         "low": round(float(lo), 4),
                         "close": round(float(c), 4)})
        mc = m.get("mc") or {}
        kd = m.get("kademe") or {}
        dil = [{"seviye": round(float(x["seviye"]), 4),
                "label": f"D{i+1}"}
               for i, x in enumerate(kd.get("dilimler") or [])
               if x.get("seviye")]
        def _f(v):
            return round(float(v), 4) if v else None
        veri = {"ohlc": ohlc,
                "bant": {"p25": _f(mc.get("p25")),
                         "p75": _f(mc.get("p75")),
                         "fair": _f(m.get("fair"))},
                "dilimler": dil,
                "izleme": _f(kd.get("izleme_esigi")),
                "cur": m.get("cur") or "$"}
        return _sahne(_FIYAT_JS, veri, 466)
    except Exception:
        return False


def _tema_koyu_mu():
    """GERÇEKTE render edilen temayı tespit eder (config değil, ekran):
    1) st.context.theme.type (v1.46+ — izleyici bazında doğru kaynak),
    2) yoksa config theme.base,
    3) o da yoksa AÇIK varsay — karanlığa BOYAMA (güvenli taraf).
    Bu, config.toml eksik/yanlış yoldayken koyu CSS'in açık temanın
    üstüne binip yazıları okunmaz yapmasını (16 Ağu vakası) önler."""
    try:
        tip = getattr(getattr(st, "context", None), "theme", None)
        tip = getattr(tip, "type", None)
        if tip in ("light", "dark"):
            return tip == "dark"
    except Exception:
        pass
    try:
        b = st.get_option("theme.base")
        if b in ("light", "dark"):
            return b == "dark"
    except Exception:
        pass
    return False


def _site_giydir():
    """PREMIUM design system — YALNIZ görünüm (CSS). Token'lar tema
    tabanına göre seçilir; seçiciler stabil data-testid + :has()
    (baseweb'e bağımlı DEĞİL). Seçici tutmazsa süs kaybolur, işlev
    bozulmaz. prefers-reduced-motion korumalı."""
    _koyu = _tema_koyu_mu()
    if _koyu:
        tok = dict(
            bg="#0b0e14", surface="#131722",
            border="rgba(255,255,255,0.08)",
            border2="rgba(255,255,255,0.13)",
            txt="#e7ecf3", txt2="rgba(231,236,243,0.62)",
            accent="#6366f1", accent2="#8b5cf6",
            gain="#22c55e", loss="#ef4444",
            glow1="rgba(99,102,241,0.14)", glow2="rgba(139,92,246,0.10)",
            golge="rgba(0,0,0,0.28)", sidebg="#0a0d14")
    else:
        tok = dict(
            bg="#fafaf9", surface="#ffffff",
            border="rgba(0,0,0,0.08)", border2="rgba(0,0,0,0.13)",
            txt="#1c1c1e", txt2="rgba(28,28,30,0.62)",
            accent="#4f46e5", accent2="#7c3aed",
            gain="#16a34a", loss="#dc2626",
            glow1="rgba(79,70,229,0.08)", glow2="rgba(124,58,237,0.06)",
            golge="rgba(16,24,40,0.08)", sidebg="#f4f4f2")
    st.markdown(f"""
<style>
:root {{
  --bg:{tok['bg']}; --surface:{tok['surface']};
  --border:{tok['border']}; --border2:{tok['border2']};
  --txt:{tok['txt']}; --txt2:{tok['txt2']};
  --accent:{tok['accent']}; --accent2:{tok['accent2']};
  --gain:{tok['gain']}; --loss:{tok['loss']};
  --r-sm:8px; --r-md:12px; --r-lg:16px;
  --shadow-1:0 1px 2px {tok['golge']};
  --shadow-2:0 4px 12px {tok['golge']}, 0 0 0 1px var(--border);
  --shadow-3:0 12px 32px {tok['golge']}, 0 0 0 1px var(--border2);
  --ease:cubic-bezier(0.4,0,0.2,1);
}}
#MainMenu, footer {{visibility: hidden;}}
[data-testid="stDecoration"] {{display: none;}}

/* Katmanlı glow arka plan */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 600px at 15% -10%, {tok['glow1']}, transparent 60%),
    radial-gradient(1000px 500px at 90% 0%, {tok['glow2']}, transparent 55%),
    var(--bg);
  background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.28;
  background-image: radial-gradient(var(--border) 1px, transparent 1px);
  background-size: 22px 22px; z-index:0;
}}

/* Yapışkan cam header */
.stAppHeader, [data-testid="stHeader"] {{
  background: transparent !important;
  -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--border);
}}

/* İçerik: genişlik + yumuşak giriş */
.stMainBlockContainer, .block-container {{
  padding-top: 3rem; padding-bottom: 3rem; max-width: 1280px;
  animation: fadeUp .45s var(--ease) both;
}}
@keyframes fadeUp {{
  from {{opacity:0; transform:translateY(8px);}}
  to {{opacity:1; transform:none;}}
}}

/* Başlıklar: sıkı + gradient h1 */
h1, h2 {{letter-spacing:-0.02em; font-weight:700;}}
h1 {{
  background: linear-gradient(135deg, var(--txt), var(--accent));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}}

/* Metric → KPI kartı (tabular rakam) */
[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 14px 18px;
  box-shadow: var(--shadow-2);
  transition: transform .18s var(--ease), box-shadow .18s var(--ease);
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-2px); box-shadow: var(--shadow-3);
}}
[data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums; font-feature-settings:"tnum" 1;
  font-weight:700; letter-spacing:-0.01em;
}}
[data-testid="stMetricLabel"] {{color: var(--txt2); font-weight:500;}}
[data-testid="stMetricDelta"] {{
  font-variant-numeric: tabular-nums; border-radius:999px;
  padding:2px 8px; font-weight:600; font-size:.82rem; width:fit-content;
}}

/* container(border=True) → kart */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--surface);
  border:1px solid var(--border) !important;
  border-radius: var(--r-lg); box-shadow: var(--shadow-2);
}}

/* Expander kart */
[data-testid="stExpander"] {{
  border:1px solid var(--border); border-radius: var(--r-md);
  background: var(--surface); overflow:hidden;
}}
[data-testid="stExpander"] details > summary {{
  font-weight:600; transition: background .15s var(--ease);
}}
[data-testid="stExpander"] details > summary:hover {{
  background: color-mix(in srgb, var(--accent) 8%, transparent);
}}

/* Alert'ler: zarif callout + tür aksanı */
.stAlert {{
  border-radius: var(--r-md) !important;
  border:1px solid var(--border) !important;
  box-shadow: var(--shadow-1);
}}
.stAlert[kind="info"]    {{border-left:4px solid var(--accent) !important;}}
.stAlert[kind="success"] {{border-left:4px solid var(--gain) !important;}}
.stAlert[kind="warning"] {{border-left:4px solid #f59e0b !important;}}
.stAlert[kind="error"]   {{border-left:4px solid var(--loss) !important;}}

/* Sidebar zemini — config uygulanmasa bile tutarlı kalır */
[data-testid="stSidebar"] {{
  background: {tok['sidebg']};
  border-right: 1px solid var(--border);
}}

/* Sidebar: premium nav pill (baseweb'siz :has ile) */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
  display:flex; align-items:center; padding:8px 12px; margin:2px 0;
  border-radius: var(--r-sm); cursor:pointer;
  transition: background .15s var(--ease);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: linear-gradient(90deg,
      color-mix(in srgb, var(--accent) 22%, transparent), transparent);
  box-shadow: inset 3px 0 0 var(--accent);
  font-weight:600; color: var(--txt);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child
  {{display:none;}}

/* Butonlar */
[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border:none; color:#fff; font-weight:600;
  box-shadow: 0 2px 8px color-mix(in srgb, var(--accent) 40%, transparent);
  transition: transform .15s var(--ease), box-shadow .15s var(--ease);
}}
[data-testid="stBaseButton-primary"]:hover {{
  transform: translateY(-1px);
  box-shadow: 0 6px 18px color-mix(in srgb, var(--accent) 50%, transparent);
}}
[data-testid="stBaseButton-secondary"] {{
  background: transparent; border:1px solid var(--border2);
  color: var(--txt); font-weight:500;
  transition: background .15s var(--ease), border-color .15s var(--ease);
}}
[data-testid="stBaseButton-secondary"]:hover {{
  background: color-mix(in srgb, var(--accent) 8%, transparent);
  border-color: var(--accent);
}}

/* Tabs (baseweb kaldırılırsa yalnız süs gider) */
.stTabs [data-baseweb="tab-list"] {{gap:4px; border-bottom:1px solid var(--border);}}
.stTabs [data-baseweb="tab"] {{font-weight:500; color:var(--txt2);}}
.stTabs [aria-selected="true"] {{color:var(--txt); font-weight:600;}}

/* Input odak halkası */
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 30%, transparent);
}}

/* Dataframe dış çerçeve (canvas içi stillenemez — mimari sınır) */
[data-testid="stDataFrame"] {{
  border-radius: var(--r-md); overflow:hidden; box-shadow: var(--shadow-2);
}}

/* Özel scrollbar */
::-webkit-scrollbar {{width:10px; height:10px;}}
::-webkit-scrollbar-track {{background: transparent;}}
::-webkit-scrollbar-thumb {{
  background: var(--border2); border-radius:999px;
  border:2px solid transparent; background-clip: padding-box;
}}
::-webkit-scrollbar-thumb:hover {{background: var(--accent);}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{animation:none !important; transition:none !important;}}
}}
</style>""", unsafe_allow_html=True)
    T = _ui_tok()
    _koyu2 = _tema_koyu_mu()
    _orb1 = "rgba(99,102,241,0.20)" if _koyu2 else "rgba(79,70,229,0.10)"
    _orb2 = "rgba(34,197,94,0.10)" if _koyu2 else "rgba(22,163,74,0.07)"
    st.markdown(f"""
<style>
/* ═══ VİTRİN KATMANI v2 — iskelet + hareket + imza tipografi ═══ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]
  {{display:none !important;}}
::selection {{background:{T['accent']}55;}}

/* Nefes alan zemin: iki yavaş ışık küresi + tanecik dokusu */
@keyframes orbA {{0%{{transform:translate(0,0)}}50%{{transform:translate(70px,40px)}}100%{{transform:translate(0,0)}}}}
@keyframes orbB {{0%{{transform:translate(0,0)}}50%{{transform:translate(-60px,-30px)}}100%{{transform:translate(0,0)}}}}
[data-testid="stAppViewContainer"]::after {{
  content:""; position:fixed; width:640px; height:640px; left:-160px;
  top:-180px; border-radius:50%; pointer-events:none; z-index:0;
  background:radial-gradient(circle,{_orb1},transparent 62%);
  animation:orbA 34s ease-in-out infinite; filter:blur(10px);}}
.stMainBlockContainer::after {{
  content:""; position:fixed; width:560px; height:560px; right:-140px;
  bottom:-160px; border-radius:50%; pointer-events:none; z-index:0;
  background:radial-gradient(circle,{_orb2},transparent 60%);
  animation:orbB 41s ease-in-out infinite; filter:blur(10px);}}
[data-testid="stAppViewContainer"]::before {{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.5;
  z-index:0; background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><filter id='n'><feTurbulence baseFrequency='0.9' numOctaves='2'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.017 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");}}

/* İmza tipografi: başlıklar Space Grotesk */
h1,h2,h3,.hero h1,.gauge-say,.vb-l {{font-family:'Space Grotesk','Inter',sans-serif;}}

/* HERO v2: eyebrow + dev gradyan başlık + gradyan çizgi */
.hero {{position:relative;background:
       linear-gradient(120deg,{T['accent']}24,transparent 55%),{T['card']};
       border:1px solid {T['line']};border-radius:20px;
       padding:26px 30px 22px;margin:4px 0 18px;overflow:hidden;
       box-shadow:0 20px 50px -30px {T['accent']}66;}}
.hero::after {{content:"";position:absolute;inset:0;pointer-events:none;
       background:repeating-linear-gradient(115deg,transparent 0 22px,
       {T['line']} 22px 23px);opacity:.22;}}
.hero-eyebrow {{color:{T['accent']};font-size:.72rem;font-weight:700;
       letter-spacing:.22em;text-transform:uppercase;margin-bottom:8px;}}
.hero h1 {{margin:0;font-size:2.1rem;line-height:1.08;font-weight:700;
       letter-spacing:-.025em;background:linear-gradient(100deg,
       {T['txt']},{T['accent']} 130%);
       -webkit-background-clip:text;background-clip:text;color:transparent;}}
.hero p {{margin:8px 0 0;color:{T['mut']};font-size:.95rem;}}
.hero-cizgi {{height:2px;margin-top:16px;border-radius:2px;
       background:linear-gradient(90deg,{T['accent']},{T['accent']}00 70%);}}

/* KPI v2: giriş animasyonu + üst aksan + dev ilk kart + spark */
@keyframes kpiIn {{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
.kpi-grid {{display:grid;gap:13px;margin:6px 0 16px;
       grid-template-columns:repeat(auto-fit,minmax(158px,1fr));}}
.kpi {{position:relative;background:{T['card']};border:1px solid {T['line']};
       border-radius:16px;padding:14px 16px 12px;overflow:hidden;
       box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
       animation:kpiIn .5s cubic-bezier(.4,0,.2,1) both;
       transition:transform .18s,box-shadow .18s,border-color .18s;}}
.kpi:nth-child(1){{animation-delay:.02s}} .kpi:nth-child(2){{animation-delay:.08s}}
.kpi:nth-child(3){{animation-delay:.14s}} .kpi:nth-child(4){{animation-delay:.20s}}
.kpi:nth-child(5){{animation-delay:.26s}} .kpi:nth-child(6){{animation-delay:.32s}}
.kpi::before {{content:"";position:absolute;top:0;left:14px;right:14px;
       height:2px;border-radius:2px;background:var(--ka,{T['accent']});
       opacity:.85;}}
.kpi:hover {{transform:translateY(-3px);border-color:{T['line']};
       box-shadow:0 14px 34px -18px var(--ka,{T['accent']});}}
.kpi-l {{color:{T['mut']};font-size:.7rem;text-transform:uppercase;
       letter-spacing:.09em;font-weight:700;}}
.kpi-v {{color:{T['txt']};font-size:1.42rem;font-weight:750;margin-top:3px;
       font-variant-numeric:tabular-nums;letter-spacing:-.02em;}}
.kpi:first-child .kpi-v {{font-size:1.9rem;background:linear-gradient(100deg,
       {T['txt']},var(--ka,{T['accent']}));-webkit-background-clip:text;
       background-clip:text;color:transparent;}}
.spark {{display:block;margin-top:6px;}}

/* GAUGE v2: süpürerek dolan halka (+@property) + ışıma */
@property --gp {{syntax:'<angle>';initial-value:0deg;inherits:false;}}
@keyframes gsweep {{from{{--gp:0deg}}}}
.gauge-kap {{display:flex;flex-direction:column;align-items:center;gap:7px;margin:6px 0;}}
.gauge-ring {{width:122px;height:122px;border-radius:50%;position:relative;
       background:radial-gradient(closest-side,{T['card']} 72%,transparent 73% 100%),
       conic-gradient(from -90deg,var(--rk) var(--gp),{T['line']} 0);
       display:flex;align-items:center;justify-content:center;
       border:1px solid {T['line']};animation:gsweep 1.1s cubic-bezier(.3,.7,.3,1) both;
       box-shadow:0 0 34px -10px var(--rk);}}
.gauge-say {{color:{T['txt']};font-weight:700;font-size:1.7rem;letter-spacing:-.02em;}}
.gauge-b {{color:{T['mut']};font-size:.72rem;margin-left:2px;}}
.gauge-l {{color:{T['mut']};font-size:.8rem;letter-spacing:.03em;}}

/* HÜKÜM banner v2 */
.vb {{position:relative;display:flex;gap:16px;align-items:center;
      border:1px solid {T['line']};border-radius:18px;overflow:hidden;
      padding:18px 22px;margin:8px 0 12px;background:
      linear-gradient(95deg,color-mix(in srgb,var(--vk) 16%,transparent),transparent 65%),{T['card']};
      box-shadow:0 18px 44px -26px var(--vk);}}
.vb::after {{content:"";position:absolute;inset:0;pointer-events:none;opacity:.16;
      background:repeating-linear-gradient(115deg,transparent 0 16px,var(--vk) 16px 17px);}}
.vb-e {{flex:0 0 auto;width:56px;height:56px;border-radius:16px;display:flex;
      align-items:center;justify-content:center;font-size:1.7rem;
      background:color-mix(in srgb,var(--vk) 18%,transparent);border:1px solid color-mix(in srgb,var(--vk) 40%,transparent);}}
.vb-l {{color:{T['txt']};font-weight:700;font-size:1.32rem;letter-spacing:-.015em;}}
.vb-m {{color:{T['mut']};font-size:.93rem;margin-top:3px;max-width:70ch;}}

/* Butonlar: parlama süpürmesi */
[data-testid="stBaseButton-primary"] {{position:relative;overflow:hidden;}}
[data-testid="stBaseButton-primary"]::after {{content:"";position:absolute;top:0;
   left:-70%;width:45%;height:100%;transform:skewX(-20deg);
   background:linear-gradient(90deg,transparent,rgba(255,255,255,.34),transparent);
   transition:left .5s ease;}}
[data-testid="stBaseButton-primary"]:hover::after {{left:125%;}}

@media (prefers-reduced-motion: reduce) {{
  .kpi,.gauge-ring,[data-testid="stAppViewContainer"]::after,
  .stMainBlockContainer::after {{animation:none !important;}}
}}
</style>""", unsafe_allow_html=True)



def _giris_kapisi():
    """İnternete açık kurulum için OPSİYONEL şifre kapısı.
    Şifre; ortam değişkeni APP_SIFRE ya da .streamlit/secrets.toml
    içindeki APP_SIFRE ile verilir. İKİSİ DE YOKSA kapı tamamen
    DEVRE DIŞIDIR — yerelde hiçbir şey değişmez."""
    sifre = os.environ.get("APP_SIFRE")
    if not sifre:
        try:
            sifre = st.secrets.get("APP_SIFRE")
        except Exception:
            sifre = None
    if not sifre:
        return
    if st.session_state.get("_giris_ok"):
        return
    st.markdown("## 🔒 Giriş")
    st.caption("Bu kurulum şifre korumalı (APP_SIFRE tanımlı).")
    g = st.text_input("Şifre", type="password", key="_giris_sifre")
    if st.button("Gir", type="primary"):
        if g == str(sifre):
            st.session_state["_giris_ok"] = True
            st.rerun()
        st.error("Şifre yanlış.")
    st.stop()


def _pg_tek():
    _hero("Tek Hisse — Derin Analiz",
          "Ham veri → matematik → hüküm → istersen Claude raporu",
          ust="🔬 Analiz Terminali")
    g = render_girdiler()
    _tek_hisse_akisi(*g)


def _pg_tarama():
    render_screener()


def _pg_rakip():
    render_rakip()


def _pg_portfoy():
    render_portfoy()


def _pg_piyasa():
    render_piyasa()


def _pg_kripto():
    render_kripto()


def _sayfalar():
    """Üst navigasyon sayfaları — eski MOD radio'nun yerini alır."""
    return {
        "tek": st.Page(_pg_tek, title="Tek Hisse", icon="🔬",
                       url_path="tek", default=True),
        "tarama": st.Page(_pg_tarama, title="Yaklaşanlar", icon="🎯",
                          url_path="tarama"),
        "rakip": st.Page(_pg_rakip, title="Rakip Kıyası", icon="⚔️",
                         url_path="rakip"),
        "portfoy": st.Page(_pg_portfoy, title="Portföy", icon="👜",
                           url_path="portfoy"),
        "piyasa": st.Page(_pg_piyasa, title="Piyasa", icon="📉",
                          url_path="piyasa"),
        "kripto": st.Page(_pg_kripto, title="Kripto", icon="🪙",
                          url_path="kripto"),
    }


def main():
    st.set_page_config(
        page_title=APP_TITLE, page_icon="🔍", layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None,
                    "About": (APP_TITLE + " — kişisel analiz "
                              "terminali. Yatırım tavsiyesi "
                              "değildir.")})
    _site_giydir()
    _giris_kapisi()

    # Tarama/Rakip ekranlarından gelen yönlendirmeyi widget'lar
    # çizilmeden UYGULA (Streamlit: widget oluştuktan sonra key'ine
    # yazılamaz — bu yüzden buton yalnız not bırakır, burada işlenir).
    if "_git_tkr" in st.session_state:
        st.session_state["tkr"] = st.session_state.pop("_git_tkr")
        st.session_state.pop("_git_mod", None)
        st.session_state["_git_tek"] = True

    # ── ÜST NAVİGASYON KABUĞU — kenar çubuğu ve mod radio kaldırıldı ─
    sayfalar = _sayfalar()
    nav = st.navigation(list(sayfalar.values()), position="top")
    if st.session_state.pop("_git_tek", False):
        st.switch_page(sayfalar["tek"])
    nav.run()


def _tek_hisse_akisi(ticker, run, a, api_key, model, use_web, edu,
                     use_edgar, peers, call_text, maliyet):
    """Tek Hisse akışı — eski main() gövdesi (davranış birebir aynı)."""
    if run and ticker:
        try:
            with st.spinner(f"{ticker} verisi indiriliyor…"):
                st.session_state["bundle"] = fetch_bundle(ticker, use_edgar)
            # DÜZELTME: Streamlit kuralı — bir widget (key="tkr") aynı
            # çalıştırmada çizildikten SONRA o key'e doğrudan yazılamaz
            # (StreamlitAPIException). Bu yüzden buraya sadece not bırakılır;
            # main() başındaki köprü, rerun'da widget'lar çizilmeden önce
            # değeri "tkr"ye güvenle uygular.
            st.session_state["_git_tkr"] = ticker
            try:
                st.session_state["sugg"] = suggest_assumptions(
                    st.session_state["bundle"])
            except Exception:
                st.session_state["sugg"] = None
            st.session_state.pop("claude_report", None)
            st.session_state.pop("claude_meta", None)
            # K2 DÜZELTMESİ: st.rerun() bu koşuyu anında keser ve rerun'da
            # buton durumu (run) False döner — aşağıdaki günlük bloğu bu
            # yüzden HİÇ çalışmıyordu. Bayrak bırakılır, rerun sonrası
            # metrikler hesaplandıktan sonra tüketilir.
            st.session_state["_kayit_bekliyor"] = True
            st.rerun()   # kenar çubuğu önerileri yeni şirkete göre yenilensin
        except Exception as e:
            st.session_state.pop("bundle", None)
            st.error(str(e))

    b = st.session_state.get("bundle")
    if not b:
        st.markdown(f"# 🔍 {APP_TITLE}")
        st.markdown(
            "Sol taraftan bir **ticker** gir ve **Analizi Çalıştır** de.\n\n"
            "Hat şu sırayla işler: yfinance ham verisi → oran/kalite motoru → "
            "DCF + **ters-DCF** → pahalılık skoru ve hüküm → istersen Claude "
            "derin analiz raporu.\n\n"
            "*Bu araç eğitim/araştırma amaçlıdır; yatırım tavsiyesi değildir.*")
        return

    try:
        m = compute_metrics(b, a)
    except Exception:
        st.error("Metrik motoru bu ticker'da beklenmeyen bir veriye çarptı. "
                 "Ayrıntı aşağıda — bu çıktıyı paylaşırsan hattı güçlendiririz.")
        with st.expander("Teknik ayrıntı"):
            st.code(traceback.format_exc())
        return

    m["edu"] = edu
    m["call"] = analyze_call_text(call_text)
    m["cikis"] = cikis_plani(m, maliyet if maliyet > 0 else None)
    # 📓 Kayıt defteri: önce geçmişi oku (bu ticker'ın son 3 kaydı),
    # sonra bu çalıştırmayı günlüğe yaz (yalnız butona basıldığında).
    try:
        gecmis = [k for k in gunluk_oku()
                  if k.get("ticker") == m["ticker"]][-3:]
    except Exception:
        gecmis = []
    m["gecmis"] = gecmis or None
    # K2 DÜZELTMESİ: 'if run:' rerun sonrası daima False'tu (Streamlit
    # buton kuralı) → karar günlüğü hiç yazılmıyor, kalibrasyon zinciri
    # yapısal olarak boş kalıyordu. Fetch yolunda bırakılan bayrak
    # burada tüketilir; fetch başarısızsa bayrak da yoktur (doğru).
    if st.session_state.pop("_kayit_bekliyor", False):
        kd = m.get("kademe") or {}
        # ══ B8-1 (KRİTİK — B2-1 ile aynı sınıf) ══════════════════════
        # main() bu alanları yazıyordu ama kayit_coz() `tip`/`ufuk_bitis`/
        # `deger_cipa`, kalibrasyon() ise `tip`/`olasilik`/`sonuc` arıyordu.
        # `tip` alanı HİÇ YAZILMADIĞI için 50 kayıtlık simülasyonda
        # kayit_coz() 0 çözülebilir buluyor, kalibrasyon() None dönüyordu:
        # "Ufku dolan kararları çöz" butonu HER ZAMAN "karar yok" diyor,
        # Brier skoru ve bant tablosu ASLA görünmüyordu. Kullanıcı ekranda
        # "Kalibrasyon" başlığını görüyor ama o başlık YAPISAL OLARAK
        # dolamıyordu.
        _cipa = (kd.get("izleme_esigi")
                 or ((kd.get("dilimler") or [{}])[0]).get("seviye"))
        _ols = None
        try:
            _ols = _num(st.session_state.get("olasilik_in"))
        except Exception:
            _ols = None
        gunluk_yaz({"tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "ticker": m["ticker"], "fiyat": m.get("price"),
                    "skor": m.get("score"), "kalite": m.get("quality"),
                    "hukum": (m.get("verdict") or ["?"])[0][:60],
                    "adil_baz": m.get("fair"),
                    "esik_veya_dilim1": _cipa,
                    "mc_p50": (m.get("mc") or {}).get("p50"),
                    # ── çözücünün beklediği ŞEMA ────────────────────────
                    "tip": ("karar" if _ols is not None else "gozlem"),
                    "cozuldu": False,
                    "ufuk": (m.get("ufuk") or {}).get("ufuk"),
                    "ufuk_bitis": (m.get("ufuk") or {}).get(
                        "zaman_stopu_tarihi"),
                    "deger_cipa": _cipa,
                    "olasilik": _ols,
                    "cozum_kurali": "bitis",
                    "premortem": st.session_state.get("premortem_in")})
    m["peers"] = None
    if peers:
        try:
            m["peers"] = build_peer_table(m, fetch_peers(peers))
        except Exception:
            m["peers"] = None
    try:
        _spk = _sparkline_svg(
            list((m.get("_hist") or {}).get("Close", [])),
            _ui_tok()["accent"])
    except Exception:
        _spk = ""
    _kpi = [("Fiyat",
             f"{m['cur']}{m['price']:,.2f}" if m["price"] else "—",
             "son 60 gün" if _spk else None, "accent", _spk)]
    # (a3) Tek-nokta DCF yerine MONTE CARLO BANDI birincil: aralık, sahte
    # kesinlikten dürüsttür. Sıra: MC P25–P75 → senaryo aralığı → tek nokta.
    _mch = m.get("mc") or {}
    if _mch.get("p25") and _mch.get("p75"):
        fair_txt = f"{m['cur']}{_mch['p25']:,.0f}–{_mch['p75']:,.0f}"
        fair_lbl = "Değer Bandı (MC P25–P75)"
    elif m.get("fair_lo") and m.get("fair_hi"):
        fair_txt = f"{m['cur']}{m['fair_lo']:,.0f}–{m['fair_hi']:,.0f}"
        fair_lbl = "Adil Değer (senaryo aralığı)"
    elif m["fair"]:
        # ── B11-7 ────────────────────────────────────────────────────
        # Monte Carlo bandı ±%30 açıkken adil değeri "176.59" diye iki
        # ondalıkla yazmak SAHTE KESİNLİKTİR. Witteman vd. (JMIR 2011):
        # ondalıklı sayı, hak edilmemiş güvenilirlik sinyali verir.
        # Nokta tahmin, kendi belirsizliğinin genişliğine yuvarlanır.
        _gen = None
        if _mch.get("p5") and _mch.get("p95"):
            _gen = abs(_mch["p95"] - _mch["p5"])
        _adim = (10.0 if (_gen or 0) > 60 else 5.0 if (_gen or 0) > 25
                 else 1.0 if (_gen or 0) > 5 else 0.01)
        _yuv = round(m["fair"] / _adim) * _adim
        fair_txt = (f"{m['cur']}{_yuv:,.0f}" if _adim >= 1
                    else f"{m['cur']}{_yuv:,.2f}")
        fair_lbl = "Adil Değer (tek nokta — temkinli)"
    else:
        fair_txt, fair_lbl = "—", "Adil Değer"
    _kpi.append((fair_lbl, fair_txt, "aralık birincildir", "info"))
    _kpi.append(("Güvenlik Marjı", fmt_pct(m["mos"]), None,
                 "ok" if (m.get("mos") or 0) > 0 else "bad"))
    _kpi.append(("Kalite Skoru",
                 f"{m['quality']:.0f}/100"
                 if m.get("quality") is not None else "—", None, "ok"))
    _kpi.append(("Piyasa Değeri", fmt_money(m["mcap"], m["cur"]),
                 None, "accent"))
    if not sahne_hero(m, fair_lbl, fair_txt):
        _hero(f"{m['name']} · {m['ticker']}",
              f"{m['sector']} — {m['industry']}",
              ust="Derin Analiz Raporu")
        kpi_serit(_kpi)
    sahne_fiyat(m)

    try:
        m["_fark"] = durum_farki(m)
        durum_yaz(m.get("ticker") or "?", anlik_durum(m))
    except Exception:
        m["_fark"] = None
    render_karar_ekrani(m, a)

    t1, t2, t3, t4, t5 = st.tabs(["📐 Değerleme", "📊 Finansallar",
                                  "🚩 Risk Radarı", "📅 Katalizör & Akış",
                                  "🧠 Claude Raporu"])
    with t1:
        render_valuation_tab(m, a, edu)
    with t2:
        render_financials_tab(m, b, edu)
    with t3:
        render_risk_tab(m, edu)
    with t4:
        render_flow_tab(m, edu)
    with t5:
        render_claude_tab(m, api_key, model, use_web)

    _tz = m.get("tazelik") or {}
    _sd = _tz.get("son_dosyalama") or {}
    _sd_txt = (f" · Son SEC dosyalaması: {_sd['form']} {_sd['tarih']}"
               if _sd else "")
    st.caption(f"Veri: Yahoo (fiyat ~15 dk gecikmeli olabilir) + SEC EDGAR "
               f"birincil temel kalemler · çekilme: {b['fetched_at']}"
               f"{_sd_txt} · Skor sezgiseldir, Claude katmanı denetler · "
               f"**Yatırım tavsiyesi değildir.**")


if __name__ == "__main__":
    main()
