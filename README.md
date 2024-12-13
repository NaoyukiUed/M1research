# 環境構築
まず、このプロジェクトを管理するディレクトリを作成し、そのディレクトリに移動する。
そのディレクトリ上で以下のコードを実行する。

```
git clone https://github.com/NaoyukiUed/M1research.git
```
M1researchというディレクトリが作成されたことを確認する。
以下のコードを実行する
```
cd M1research/book_project
python -m venv myenv
```
book_project直下にmyenvディレクトリができたことを確認する。
以下のコードを実行する
```
myenv/Scripts/activate
```
ターミナルがmyenv環境になったことを確認する。
以下のコードを実行する
```
pip install -r requirements.txt
```
エラーメッセージが発生していないか確認する。
book_project直下に.envという名前のファイルを作成し、その中身を以下のようにする
```
OPENAI_API_KEY='OPENAIのAPIKEY'
```
以下のコードを実行する
```
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
ターミナルにエラーメッセージがなければ[http://127.0.0.1:8000/test/pdfs/](http://127.0.0.1:8000/test/pdfs/)にアクセスする。
右上のアップロードボタンを押し、システムで用いるPDFファイルをアップロードする。
アップロードされたPDF一覧からアップロードしたPDFを選び、詳細を見るを選択する。
右側の対話スペースから、PDFを読んで気になったことや分からなかったことに関してAIと対話する。
