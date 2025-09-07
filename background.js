chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkFakeNews",
    title: "Check with NewsScope",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === "checkFakeNews" && info.selectionText) {
    chrome.storage.local.set({ selectedText: info.selectionText }, () => {
      chrome.windows.create({
        url: chrome.runtime.getURL("popup.html"),
        type: "popup",
        width: 800,
        height: 700
      });
    });
  }
});