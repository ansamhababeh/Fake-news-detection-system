import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.*;
import java.io.FileWriter;
import java.io.PrintWriter;
import org.json.JSONArray;
import org.json.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


public class Main {
    public static void main(String[] args) {
        try {
            String apiKey = "7e52a84f16ab45a79d52d44610f81c0d";


            List<String> sources = Arrays.asList("cnn", "al-jazeera-english", "bbc-news", "associated-press");


            List<News> newsList = new ArrayList<>();

            for (String source : sources) {
                String urlStr = "https://newsapi.org/v2/top-headlines?sources=" + source + "&language=en&apiKey=" + apiKey;

                try {
                    URL url = new URL(urlStr);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("GET");

                    BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    StringBuilder response = new StringBuilder();
                    String inputLine;

                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();

                    JSONObject jsonResponse = new JSONObject(response.toString());
                    JSONArray articles = jsonResponse.getJSONArray("articles");

                    // ✅ طباعة عدد المقالات المسترجعة
                    System.out.println("🔹 " + source + ": " + articles.length() + " articles");

                    for (int i = 0; i < articles.length(); i++) {
                        JSONObject article = articles.getJSONObject(i);

                        String title = article.optString("title", "No Title");
                        String content = article.optString("content", "No Content");
                        String publishedAtStr = article.optString("publishedAt", "");
                        String urlToArticle = article.optString("url", "No URL");
                        String imageUrl = article.optString("urlToImage", "No Image");

                        String sourceName = "Unknown";
                        if (article.has("source") && !article.isNull("source")) {
                            JSONObject sourceObj = article.getJSONObject("source");
                            sourceName = sourceObj.optString("name", "Unknown");
                        }

                        Date publishDate = null;
                        try {
                            publishDate = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.ENGLISH).parse(publishedAtStr);
                        } catch (Exception e1) {
                            try {
                                publishDate = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ENGLISH).parse(publishedAtStr);
                            } catch (Exception e2) {
                                System.out.println("⚠️ Date parsing failed for: " + publishedAtStr);
                            }
                        }

                        // محاولة جلب النص الكامل
                        String fullArticle = content;
                        try {
                            Document doc = Jsoup.connect(urlToArticle).userAgent("Mozilla").get();
                            Elements paragraphs = doc.select("div.article__content p, div.l-container p, div.wysiwyg.wysiwyg--all-content p");

                            StringBuilder fullArticleBuilder = new StringBuilder();
                            for (Element paragraph : paragraphs) {
                                fullArticleBuilder.append(paragraph.text()).append("\n\n");
                            }

                            if (fullArticleBuilder.length() > 0) {
                                fullArticle = fullArticleBuilder.toString().trim();
                            }

                        } catch (Exception e) {
                            System.out.println("⚠️ Failed to fetch full content from: " + urlToArticle);
                        }

                        News news = new News(title, fullArticle, publishDate, urlToArticle, sourceName, imageUrl);
                        newsList.add(news);
                    }

                } catch (Exception e) {
                    System.out.println("❌ Failed to fetch from source: " + source);
                    e.printStackTrace();
                }
            }

            // طباعة الأخبار
            for (News news : newsList) {
                System.out.println(news);
            }

            // حفظ الأخبار بملف CSV
            try (PrintWriter writer = new PrintWriter(new FileWriter("news_data.csv"))) {
                writer.println("Title,PublishedDate,Source,URL,ImageURL,Article");

                for (News news : newsList) {
                    String row = String.format("\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"",
                            news.getTitle().replace("\"", "'"),
                            news.getPublishDate(),
                            news.getSource().replace("\"", "'"),
                            news.getUrl(),
                            news.getImageUrl(),
                            news.getArticle().replace("\"", "'").replace("\n", " ").replace("\r", " ")
                    );
                    writer.println(row);
                }

                System.out.println("✅ Combined news saved to news_data.csv");

            } catch (Exception e) {
                System.out.println("❌ Failed to write to CSV file");
                e.printStackTrace();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
