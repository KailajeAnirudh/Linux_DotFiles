function isRegexPattern(str) {
  try {
    new RegExp(str);
    return true;
  } catch (error) {
    return false;
  }
}

function includesStringOrRegex(element, longString) {
    if (longString.includes(element)) {
        return true;
    }
    
    if (isRegexPattern(element)) {
        // If the element is a RegExp, test the long string with the regex
        const regex = new RegExp(element, "i");
        return regex.test(longString);
    }

    // If the element is neither a string nor a RegExp, return false
    return false;
}

// MARK: - Content Blocker
// Functions will check content and try to block

function filterComment(element, commentRules, channelsRules) {
    
    const commentText = element.querySelector(".comment-content");
    
    var commentID;
    var channelID;
    
    const links = element.querySelectorAll("a");
    for (const link of links) {
        const href = link.getAttribute("href");
        if (href) {
            if (extractCommentID(href)) {
                commentID = extractCommentID(href).toLowerCase();
            } else if (extractChannelId(href)) {
                channelID = extractChannelId(href).toLowerCase();
            }
        }
    }
    
    // It need to remove main comment with all replies inside
    function tryToRemoveCommentWrapper() {
        if (element.parentNode.tagName.toLowerCase() === 'ytd-comment-thread-renderer') {
            element.parentNode.style.display = "none";
        } else {
            element.style.display = "none";
        }
    }
    
    // Loop comment rules
    // Check commentID and commentText (keyword, regex)
    
    for (const commentRule of commentRules) {
        const lowercasedRule = commentRule.toLowerCase();
        
        if (commentID == lowercasedRule) {
            tryToRemoveCommentWrapper();
            return;
        }
        
        if (commentText) {
            const commentTextSafe = commentText;
            
            const ariaLabel = commentTextSafe.getAttribute("aria-label");
            
            if (ariaLabel) {
                const lowercasedAria = ariaLabel.toLowerCase();
                
                if (includesStringOrRegex(lowercasedRule, lowercasedAria)) {
                    tryToRemoveCommentWrapper();
                    return;
                }
            }
            
            
        }
    }
    
    // Loop channels rules
    // Сверяю channelID and channelName
    
    for (const channelsRule of channelsRules) {
        const lowercasedRule = channelsRule.toLowerCase();
        
        if (channelID == lowercasedRule) {
            tryToRemoveCommentWrapper();
            return;
        }
    }
}

function filterVideoCard(element, videosRules, channelsRules) {
    
    const channelName = getChannelName(element);
    const videoTitle = element.querySelector("h3.media-item-headline > span[aria-label]") || element.querySelector(".reel-item-metadata > h3.typography-body-2b[aria-label]");
    
    var videoID;
    var channelID;
    
    const links = element.querySelectorAll("a");
    for (const link of links) {
        const href = link.getAttribute("href");
        if (href) {
            if (extractVideoId(href)) {
                videoID = extractVideoId(href).toLowerCase();
            } else if (extractChannelId(href)) {
                channelID = extractChannelId(href).toLowerCase();
            }
        }
    }
    
    // Loop videos rules
    // Check videoID and video name (keyword, regex)
    
    for (const videoRule of videosRules) {
        const lowercasedRule = videoRule.toLowerCase();
        
        if (videoID == lowercasedRule) {
            element.style.display = "none";
            return;
        }
        
        if (videoTitle) {
            const videoTitleSafe = videoTitle.getAttribute("aria-label").toLowerCase();
            
            if (includesStringOrRegex(lowercasedRule, videoTitleSafe)) {
                element.style.display = "none";
                return;
            }
        }
    }
    
    // Loop channels rules
    // Check channelID and channelName
    
    for (const channelsRule of channelsRules) {
        const lowercasedRule = channelsRule.toLowerCase();
        
        if (channelID == lowercasedRule) {
            element.style.display = "none";
            return;
        }
        
        if (channelName) {
            const channelNameSafe = channelName.toLowerCase();
            
            if (channelNameSafe == lowercasedRule) {
                element.style.display = "none";
                return;
            }
        }
    }
}

function filterYouTubeContent() {
    
    browser.storage.local.get([getConst.filterChannelsRulesData,
                               getConst.filterVideosRulesData,
                               getConst.filterCommentsRulesData,
                               getConst.filterPostsRulesData,
                               getConstNotSyncing.extensionIsEnabledData], function (obj) {
        
        const extensionIsEnabled = obj[getConstNotSyncing.extensionIsEnabledData] ?? true;
        
        if (extensionIsEnabled) {
            
            // Get filter lists
            const channelsRules = obj[getConst.filterChannelsRulesData] ?? [];
            const videosRules = obj[getConst.filterVideosRulesData] ?? [];
            const commentsRules = obj[getConst.filterCommentsRulesData] ?? [];
            const postsRules = obj[getConst.filterPostsRulesData] ?? [];
            
            // Select content elements
            const selector = contentTags.map(tag => `${tag}:not([filterChecked])`).join(', ');
            const contentElements = document.querySelectorAll(selector);
            
            for (const element of contentElements) {
               
                if (element.querySelector(".comment-content")) {
                    filterComment(element,
                                  commentsRules,
                                  channelsRules);
                } else if ((element.querySelector("a[href*='watch?v']")) || (element.querySelector("a[href*='shorts/']")) || (element.tagName.toLowerCase() === "ytm-compact-channel-renderer")) {
                    filterVideoCard(element,
                                    videosRules,
                                    channelsRules);
                }
                
                element.setAttribute("filterChecked", "");
            }
        }
    });
}

// MARK: - Content Changes Observer
// It need to trigger when specific elements is appearing or loading new to recheck if need block them

const queries = contentTags.map(tag => ({
    element: `${tag}`
}));

var filterObserver = new MutationSummary({
    callback: filterYouTubeContent,
    queries: queries
});
