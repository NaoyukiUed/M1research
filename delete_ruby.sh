#!/bin/bash
iconv -f SHIFT_JIS -t UTF-8 book_cat.html | sed -E 's/<ruby><rb>([^<]+)<\/rb>.*<\/ruby>/\1/' > book_cat_utf8_noruby.txt