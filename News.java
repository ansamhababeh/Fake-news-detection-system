import java.util.Date;

public class News {

    private String title;
    private String article;
    private Date publishDate;
    private String url;
    private String source;
    private String imageUrl;

    public News(String title, String article, Date publishDate, String url, String source, String imageUrl) {
        this.title = title;
        this.article = article;
        this.publishDate = publishDate;
        this.url = url;
        this.source = source;
        this.imageUrl = imageUrl;
    }

    public String getTitle() { return title; }
    public String getArticle() { return article; }
    public Date getPublishDate() { return publishDate; }
    public String getUrl() { return url; }
    public String getSource() { return source; }
    public String getImageUrl() { return imageUrl; }

    @Override
    public String toString() {
        return "Title: " + title + "\n"
                + "Article: " + article + "\n"
                + "Published Date: " + publishDate + "\n"
                + "URL: " + url + "\n"
                + "Image: " + imageUrl + "\n"
                + "Source: " + source + "\n"
                + "------------------------------";
    }
}