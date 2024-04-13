var $jscomp = $jscomp || {};
$jscomp.scope = {};
$jscomp.checkStringArgs = function(b, g, e) {
  if (null == b) {
    throw new TypeError("The 'this' value for String.prototype." + e + " must not be null or undefined");
  }
  if (g instanceof RegExp) {
    throw new TypeError("First argument to String.prototype." + e + " must not be a regular expression");
  }
  return b + "";
};
$jscomp.ASSUME_ES5 = !1;
$jscomp.ASSUME_NO_NATIVE_MAP = !1;
$jscomp.ASSUME_NO_NATIVE_SET = !1;
$jscomp.SIMPLE_FROUND_POLYFILL = !1;
$jscomp.defineProperty = $jscomp.ASSUME_ES5 || "function" == typeof Object.defineProperties ? Object.defineProperty : function(b, g, e) {
  b != Array.prototype && b != Object.prototype && (b[g] = e.value);
};
$jscomp.getGlobal = function(b) {
  return "undefined" != typeof window && window === b ? b : "undefined" != typeof global && null != global ? global : b;
};
$jscomp.global = $jscomp.getGlobal(this);
$jscomp.polyfill = function(b, g, e, B) {
  if (g) {
    e = $jscomp.global;
    b = b.split(".");
    for (B = 0; B < b.length - 1; B++) {
      var l = b[B];
      l in e || (e[l] = {});
      e = e[l];
    }
    b = b[b.length - 1];
    B = e[b];
    g = g(B);
    g != B && null != g && $jscomp.defineProperty(e, b, {configurable:!0, writable:!0, value:g});
  }
};
$jscomp.polyfill("String.prototype.startsWith", function(b) {
  return b ? b : function(b, e) {
    var g = $jscomp.checkStringArgs(this, b, "startsWith");
    b += "";
    var l = g.length, A = b.length;
    e = Math.max(0, Math.min(e | 0, g.length));
    for (var F = 0; F < A && e < l;) {
      if (g[e++] != b[F++]) {
        return !1;
      }
    }
    return F >= A;
  };
}, "es6", "es3");
$jscomp.polyfill("Object.is", function(b) {
  return b ? b : function(b, e) {
    return b === e ? 0 !== b || 1 / b === 1 / e : b !== b && e !== e;
  };
}, "es6", "es3");
$jscomp.polyfill("Array.prototype.includes", function(b) {
  return b ? b : function(b, e) {
    var g = this;
    g instanceof String && (g = String(g));
    var l = g.length;
    e = e || 0;
    for (0 > e && (e = Math.max(e + l, 0)); e < l; e++) {
      var A = g[e];
      if (A === b || Object.is(A, b)) {
        return !0;
      }
    }
    return !1;
  };
}, "es7", "es3");
$jscomp.polyfill("String.prototype.includes", function(b) {
  return b ? b : function(b, e) {
    return -1 !== $jscomp.checkStringArgs(this, b, "includes").indexOf(b, e || 0);
  };
}, "es6", "es3");
$jscomp.polyfill("String.prototype.endsWith", function(b) {
  return b ? b : function(b, e) {
    var g = $jscomp.checkStringArgs(this, b, "endsWith");
    b += "";
    void 0 === e && (e = g.length);
    e = Math.max(0, Math.min(e | 0, g.length));
    for (var l = b.length; 0 < l && 0 < e;) {
      if (g[--e] != b[--l]) {
        return !1;
      }
    }
    return 0 >= l;
  };
}, "es6", "es3");
$jscomp.findInternal = function(b, g, e) {
  b instanceof String && (b = String(b));
  for (var B = b.length, l = 0; l < B; l++) {
    var A = b[l];
    if (g.call(e, A, l, b)) {
      return {i:l, v:A};
    }
  }
  return {i:-1, v:void 0};
};
$jscomp.polyfill("Array.prototype.find", function(b) {
  return b ? b : function(b, e) {
    return $jscomp.findInternal(this, b, e).v;
  };
}, "es6", "es3");
(function() {
  var b = window, g = !1;
  String.prototype.hashCode = function() {
    var a = 0, c;
    if (0 === this.length) {
      return a;
    }
    var f = 0;
    for (c = this.length; f < c; f++) {
      var d = this.charCodeAt(f);
      a = (a << 5) - a + d;
      a |= 0;
    }
    return a;
  };
  var e = "1 ATVPDKIKX0DER A3P5ROKL5A1OLE A3JWKAKR8XB7XF A1X6FK5RDHNB96 AN1VRQENFRJN5 A3DWYIK6Y9EEQB A1AJ19PSB66TGU A11IL2PNWYJU7H A1AT7YVPFBWXBL A3P5ROKL5A1OLE AVDBXBAVVSXLQ A1ZZFT5FULY4LN ANEGB3WVEVKZB A17D2BRD4YMT0X".split(" "), B = "optOut_crawl revealStock s_boxOfferListing s_boxType s_boxHorizontal webGraphType webGraphRange overlayPriceGraph".split(" "), l = window.opera || -1 < navigator.userAgent.indexOf(" OPR/"), A = -1 < navigator.userAgent.toLowerCase().indexOf("firefox"), F = -1 < navigator.userAgent.toLowerCase().indexOf("edge/"), 
  M = /Apple Computer/.test(navigator.vendor) && /Safari/.test(navigator.userAgent), G = !l && !A && !F && !M, Q = G ? "keepaChrome" : l ? "keepaOpera" : M ? "keepaSafari" : F ? "keepaEdge" : "keepaFirefox", ca = A ? "Firefox" : M ? "Safari" : G ? "Chrome" : l ? "Opera" : F ? "Edge" : "Unknown", J = 0, C = null, K = !1;
  try {
    K = /Android|webOS|iPhone|iPad|iPod|BlackBerry/i.test(navigator.userAgent);
  } catch (a) {
  }
  if (G) {
    try {
      chrome.runtime.sendMessage("hnkcfpcejkafcihlgbojoidoihckciin", {type:"isActive"}, null, function(a) {
        chrome.runtime.lastError || a && a.isActive && (g = !0);
      });
    } catch (a) {
    }
  }
  try {
    chrome.runtime.onUpdateAvailable.addListener(function(a) {
      chrome.runtime.reload();
    });
  } catch (a) {
  }
  var R = {}, U = 0;
  chrome.runtime.onMessage.addListener(function(a, r, f) {
    if (r.tab && r.tab.url || r.url) {
      switch(a.type) {
        case "restart":
          document.location.reload(!1);
          break;
        case "setCookie":
          chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:a.key, value:a.val, secure:!0, expirationDate:(Date.now() / 1000 | 0) + 31536E3});
          "token" == a.key ? C != a.val && 64 == a.val.length && (C = a.val, c.set("token", C), chrome.tabs.query({}, function(a) {
            try {
              a.forEach(function(a) {
                try {
                  a.url && !a.incognito && (console.log("Sending new token to tab ", a.url, C), chrome.tabs.sendMessage(a.id, {key:"updateToken", value:C}));
                } catch (da) {
                  console.log(da);
                }
              });
            } catch (S) {
              console.log(S);
            }
            document.location.reload(!1);
          })) : c.set(a.key, a.val);
          break;
        case "getCookie":
          return chrome.cookies.get({url:"https://keepa.com/extension", name:a.key}, function(a) {
            null == a ? f({value:null, install:J}) : f({value:a.value, install:J});
          }), !0;
        case "openPage":
          chrome.windows.create({url:a.url, incognito:!0});
          break;
        case "isPro":
          c.stockData ? f({value:c.stockData.pro, stockData:c.stockData}) : f({value:null});
          break;
        case "getStock":
          return c.addStockJob(a, function(d) {
            0 < d.errorCode && a.cachedStock && 430 != d.errorCode ? f(a.cachedStock) : 5 == d.errorCode || 429 == d.errorCode || 430 == d.errorCode || 9 == d.errorCode ? (9 == d.errorCode && (a.getNewId = !0), setTimeout(function() {
              c.addStockJob(a, f);
            }, 1)) : f(d);
          }), !0;
        case "getFilters":
          f({value:u.getFilters()});
          break;
        case "sendData":
          r = a.val;
          if (null != r.ratings) {
            var d = r.ratings;
            if (1000 > U) {
              if ("f1" == r.key) {
                if (d) {
                  for (var w = d.length; w--;) {
                    var h = d[w];
                    null == h || null == h.asin ? d.splice(w, 1) : (h = r.domainId + h.asin + h.ls, R[h] ? d.splice(w, 1) : (R[h] = 1, U++));
                  }
                  0 < d.length && m.sendPlainMessage(r);
                }
              } else {
                m.sendPlainMessage(r);
              }
            } else {
              R = null;
            }
          } else {
            m.sendPlainMessage(r);
          }
          f({});
          break;
        case "optionalPermissionsRequired":
          f({value:(G || A || l) && "undefined" === typeof chrome.webRequest});
          break;
        case "optionalPermissionsDenied":
          c.set("optOut_crawl", "1");
          console.log("optionalPermissionsDenied");
          f({value:!0});
          break;
        case "optionalPermissionsInContent":
          r = a.val;
          "undefined" != typeof r && (r ? (c.set("optOut_crawl", "0"), console.log("granted"), chrome.runtime.reload()) : (c.set("optOut_crawl", "1"), p.reportBug("permission denied"), console.log("denied")));
          f({value:!0});
          break;
        case "optionalPermissions":
          return "undefined" === typeof chrome.webRequest && chrome.permissions.request({permissions:["webRequest", "webRequestBlocking"]}, function(a) {
            chrome.runtime.lastError || (f({value:a}), "undefined" != typeof a && (a ? (c.set("optOut_crawl", "0"), console.log("granted"), chrome.runtime.reload()) : (c.set("optOut_crawl", "1"), p.reportBug("permission denied"), console.log("denied"))));
          }), !0;
        default:
          f({});
      }
    }
  });
  window.onload = function() {
    A ? chrome.storage.local.get(["install", "optOutCookies"], function(a) {
      a.optOutCookies && 3456E5 > Date.now() - a.optOutCookies || (a.install ? p.register() : chrome.tabs.create({url:chrome.runtime.getURL("chrome/content/onboard.html")}));
    }) : p.register();
    chrome.storage.local.get(["installTimestamp"], function(a) {
      a.installTimestamp && 12 < (a.installTimestamp + "").length ? J = a.installTimestamp : (J = Date.now(), chrome.storage.local.set({installTimestamp:J}));
    });
  };
  try {
    chrome.browserAction.onClicked.addListener(function(a) {
      c.isGuest ? chrome.tabs.create({url:c.actionUrl}) : chrome.tabs.create({url:"https://keepa.com/#!manage"});
    });
  } catch (a) {
    console.log(a);
  }
  var c = {storage:chrome.storage.local, contextMenu:function() {
    try {
      chrome.contextMenus.removeAll(), chrome.contextMenus.create({title:"View products on Keepa", contexts:["page"], id:"keepaContext", documentUrlPatterns:"*://*.amazon.com/* *://*.amzn.com/* *://*.amazon.co.uk/* *://*.amazon.de/* *://*.amazon.fr/* *://*.amazon.it/* *://*.amazon.ca/* *://*.amazon.com.mx/* *://*.amazon.es/* *://*.amazon.co.jp/* *://*.amazon.in/*".split(" ")}), chrome.contextMenus.onClicked.addListener(function(a, c) {
        chrome.tabs.sendMessage(c.id, {key:"collectASINs"}, {}, function(a) {
          "undefined" != typeof a && chrome.tabs.create({url:"https://keepa.com/#!viewer/" + encodeURIComponent(JSON.stringify(a))});
        });
      });
    } catch (a) {
      console.log(a);
    }
  }, parseCookieHeader:function(a, c) {
    if (0 < c.indexOf("\n")) {
      c = c.split("\n");
      var f = 0;
      a: for (; f < c.length; ++f) {
        var d = c[f].substring(0, c[f].indexOf(";")), w = d.indexOf("=");
        d = [d.substring(0, w), d.substring(w + 1)];
        if (2 == d.length && "-" != d[1]) {
          for (w = 0; w < a.length; ++w) {
            if (a[w][0] == d[0]) {
              a[w][1] = d[1];
              continue a;
            }
          }
          a.push(d);
        }
      }
    } else {
      if (c = c.substring(0, c.indexOf(";")), f = c.indexOf("="), c = [c.substring(0, f), c.substring(f + 1)], 2 == c.length && "-" != c[1]) {
        for (f = 0; f < a.length; ++f) {
          if (a[f][0] == c[0]) {
            a[f][1] = c[1];
            return;
          }
        }
        a.push(c);
      }
    }
  }, log:function(a) {
    p.quiet || console.log(a);
  }, iframeWin:null, operationComplete:!1, counter:0, stockInit:!1, stockRequest:[], initStock:function() {
    if (!c.stockInit && "undefined" != typeof chrome.webRequest) {
      var a = ["xmlhttprequest"], b = "*://www.amazon.com/* *://www.amazon.co.uk/* *://www.amazon.es/* *://www.amazon.nl/* *://www.amazon.com.mx/* *://www.amazon.it/* *://www.amazon.in/* *://www.amazon.de/* *://www.amazon.fr/* *://www.amazon.co.jp/* *://www.amazon.ca/* *://www.amazon.com.br/* *://www.amazon.com.au/* *://www.amazon.com.mx/* *://smile.amazon.com/* *://smile.amazon.co.uk/* *://smile.amazon.es/* *://smile.amazon.nl/* *://smile.amazon.com.mx/* *://smile.amazon.it/* *://smile.amazon.in/* *://smile.amazon.de/* *://smile.amazon.fr/* *://smile.amazon.co.jp/* *://smile.amazon.ca/* *://smile.amazon.com.br/* *://smile.amazon.com.au/* *://smile.amazon.com.mx/*".split(" ");
      try {
        var f = [c.stockData.addCartHeaders, c.stockData.geoHeaders, c.stockData.setAddressHeaders, c.stockData.addressChangeHeaders, c.stockData.productPageHeaders, c.stockData.toasterHeaders];
        chrome.webRequest.onBeforeSendHeaders.addListener(function(a) {
          if (a.initiator) {
            if (a.initiator.startsWith("http")) {
              return;
            }
          } else {
            if (a.originUrl && !a.originUrl.startsWith("moz-extension")) {
              return;
            }
          }
          var d = a.requestHeaders, h = {};
          try {
            for (var b = null, r = 0; r < d.length; ++r) {
              if ("krequestid" == d[r].name) {
                b = d[r].value;
                d.splice(r--, 1);
                break;
              }
            }
            if (b) {
              var e = c.stockRequest[b];
              c.stockRequest[a.requestId] = e;
              setTimeout(function() {
                delete c.stockRequest[a.requestId];
              }, 30000);
              var p = f[e.requestType];
              for (b = 0; b < d.length; ++b) {
                var g = d[b].name.toLowerCase();
                (p[g] || "" === p[g] || p[d[b].name] || "cookie" == g || "content-type" == g || "sec-fetch-dest" == g || "sec-fetch-mode" == g || "sec-fetch-user" == g || "accept" == g || "referer" == g) && d.splice(b--, 1);
              }
              if (0 == e.requestType && 19 > e.stockSession.length) {
                return h.cancel = !0, h;
              }
              var n = c.stockData.isMobile ? "https://" + e.host + "/gp/aw/d/" + e.asin + "/" : e.referer, l;
              for (l in p) {
                var m = p[l];
                if (0 != m.length) {
                  m = m.replace("{COOKIE}", e.stockSession).replace("{REFERER}", n).replace("{ORIGIN}", e.host);
                  if (-1 < m.indexOf("{CSRF}")) {
                    if (e.csrf) {
                      m = m.replace("{CSRF}", e.csrf), e.csrf = null;
                    } else {
                      continue;
                    }
                  }
                  d.push({name:l, value:m});
                }
              }
              for (p = 0; p < d.length; ++p) {
                var u = d[p].name.toLowerCase();
                (c.stockData.stockHeaders[u] || "" === c.stockData.stockHeaders[u] || c.stockData.stockHeaders[d[p].name] || "origin" == u || "pragma" == u || "cache-control" == u || "upgrade-insecure-requests" == u) && d.splice(p--, 1);
              }
              for (var A in c.stockData.stockHeaders) {
                var C = c.stockData.stockHeaders[A];
                0 != C.length && (C = C.replace("{COOKIE}", e.stockSession).replace("{REFERER}", n).replace("{ORIGIN}", e.host).replace("{LANG}", c.stockData.languageCode[e.domainId]), d.push({name:A, value:C}));
              }
              h.requestHeaders = d;
              a.requestHeaders = d;
            } else {
              return h;
            }
          } catch (L) {
            h.cancel = !0;
          }
          return h;
        }, {urls:b, types:a}, G ? ["blocking", "requestHeaders", "extraHeaders"] : ["blocking", "requestHeaders"]);
        chrome.webRequest.onHeadersReceived.addListener(function(a) {
          if (a.initiator) {
            if (a.initiator.startsWith("http")) {
              return;
            }
          } else {
            if (a.originUrl && !a.originUrl.startsWith("moz-extension")) {
              return;
            }
          }
          var d = a.responseHeaders, f = {};
          try {
            var b = c.stockRequest[a.requestId];
            if (b) {
              var r = b.cookies || [];
              for (a = 0; a < d.length; ++a) {
                "set-cookie" == d[a].name.toLowerCase() && (c.parseCookieHeader(r, d[a].value), d.splice(a, 1), a--);
              }
              b.cookies = r;
              switch(b.requestType) {
                case 0:
                case 1:
                case 2:
                case 4:
                case 5:
                  f.responseHeaders = d;
                  break;
                case 3:
                  f.cancel = !0, setTimeout(function() {
                    b.cookies = r;
                    c.stockSessions[b.domainId] = r;
                    b.callback();
                  }, 10);
              }
              if (0 != b.requestType) {
                d = "";
                for (a = 0; a < b.cookies.length; ++a) {
                  var e = b.cookies[a];
                  d += e[0] + "=" + e[1] + "; ";
                  "session-id" == e[0] && 16 < e[1].length && 65 > e[1].length && e[1] != b.session && (b.sessionIdMismatch = !0);
                }
                b.stockSession = d;
              }
            } else {
              return f;
            }
          } catch (ea) {
            f.cancel = !0;
          }
          return f;
        }, {urls:b, types:a}, G ? ["blocking", "responseHeaders", "extraHeaders"] : ["blocking", "responseHeaders"]);
        c.stockInit = !0;
      } catch (d) {
        p.reportBug(d, d.message + " stock exception: " + typeof chrome.webRequest + " " + ("undefined" != typeof chrome.webRequest ? typeof chrome.webRequest.onBeforeSendHeaders : "~") + " " + ("undefined" != typeof chrome.webRequest ? typeof chrome.webRequest.onHeadersReceived : "#"));
      }
    }
  }, stockData:null, isGuest:!0, actionUrl:"https://keepa.com/#!features", stockJobQueue:[], stockSessions:[], addStockJob:function(a, b) {
    a.gid = p.Guid.newGuid().substr(0, 8);
    a.requestType = -1;
    c.stockRequest[a.gid] = a;
    var f = function(a) {
      c.stockJobQueue.shift();
      b(a);
      0 < c.stockJobQueue.length && c.processStockJob(c.stockJobQueue[0][0], c.stockJobQueue[0][1]);
    };
    c.stockJobQueue.push([a, f]);
    1 == c.stockJobQueue.length && c.processStockJob(a, f);
  }, processStockJob:function(a, b) {
    if (null == c.stockData.stock) {
      console.log("stock retrieval not initialized"), b({error:"stock retrieval not initialized", errorCode:0});
    } else {
      if (0 == c.stockData.stockEnabled[a.domainId]) {
        console.log("stock retrieval not supported for domain"), b({error:"stock retrieval not supported for domain", errorCode:1});
      } else {
        if (!0 === c.stockData.pro || a.force) {
          if (a.maxQty) {
            if (!a.isMAP && c.stockData.stockMaxQty && a.maxQty < c.stockData.stockMaxQty) {
              b({stock:a.maxQty, limit:!1});
              return;
            }
            a.cachedStock = {stock:a.maxQty, limit:!1, isMaxQty:a.maxQty};
          }
          null == a.oid ? (console.log("missing oid", a), b({error:"stock retrieval failed for offer: " + a.asin + " id: " + a.gid + " missing oid.", errorCode:12})) : a.onlyMaxQty && !a.isMAP ? b() : null == a.sellerId ? b({error:"Unable to retrieve stock for this offer. ", errorCode:45}) : (c.initStock(), setTimeout(function() {
            if (c.stockInit) {
              if (setTimeout(function() {
                delete c.stockSessions[a.domainId];
              }, 36E5), setTimeout(function() {
                delete c.stockRequest[a.gid];
              }, 3E5), a.queue = [function() {
                for (var d = "", f = !1, h = !1, e = 0, r = 0; r < a.cookies.length; ++r) {
                  var g = a.cookies[r];
                  d += g[0] + "=" + g[1] + "; ";
                  "session-id" == g[0] && 16 < g[1].length && 65 > g[1].length && (f = !0, g[1] != a.session && (h = !0, e = g[1]));
                }
                a.cookie = d;
                f && h ? (a.stockSession = d, d = c.stockData.addCartUrl, f = c.stockData.addCartPOST, a.requestType = 0, p.httpPost("https://" + a.host + d.replaceAll("{SESSION_ID}", e).replaceAll("{OFFER_ID}", a.oid).replaceAll("{ADDCART}", c.stockData.stockAdd[a.domainId]).replaceAll("{ASIN}", a.asin), f.replaceAll("{SESSION_ID}", e).replaceAll("{OFFER_ID}", a.oid).replaceAll("{ADDCART}", c.stockData.stockAdd[a.domainId]).replaceAll("{ASIN}", a.asin), function(d) {
                  var f = d.match(new RegExp(c.stockData.stockV.replaceAll("{ASIN}", a.asin).replaceAll("{SELLER}", a.sellerId))), h = (new RegExp(c.stockData.limit)).test(d);
                  if (f && f[1]) {
                    d = parseInt(f[2]);
                    f = f[1].toString();
                    var w = f.includes(".") ? f.split(".")[1].length : 0;
                    f = f.replace(".", "");
                    f = parseInt(f, 10) * Math.pow(10, 2 - w);
                    b({stock:d, orderLimit:-1, limit:h, price:f});
                  } else {
                    if ((h = d.match(/automated access|api-services-support@/)) || a.isRetry) {
                      delete c.stockSessions[a.domainId], a.cookie = null, a.stockSession = null, a.cookies = null;
                    }
                    h ? (b({error:"Amazon stock retrieval rate limited (bot detection) of offer: " + a.asin + " id: " + a.gid + " offer: " + a.oid, errorCode:5}), console.log("stock retrieval rate limited for offer: ", a.asin + " " + a.oid + " id: " + a.gid + " seller.id " + a.sellerId, d.length)) : b({error:"Stock retrieval failed for this offer. Try reloading the page after a while. ", errorCode:430});
                  }
                }, !1, a.gid)) : (p.reportBug(null, "stock session issue: " + f + " " + h + " counter: " + c.counter + " c: " + JSON.stringify(a.cookies) + " " + JSON.stringify(a)), b({error:"stock session issue: " + f + " " + h, errorCode:4}));
              }], a.getNewId && (c.stockData.geoRetry && delete c.stockSessions[a.domainId], a.queue.unshift(function() {
                a.requestType = 4;
                p.httpGet("https://" + c.stockData.offerUrl.replace("{ORIGIN}", a.host).replace("{ASIN}", a.asin).replace("{SID}", a.sellerId), function(f) {
                  if (f.match(c.stockData.sellerIdBBVerify.replace("{SID}", a.sellerId))) {
                    for (var d = null, h = 0; h < c.stockData.csrfBB.length; h++) {
                      var e = f.match(new RegExp(c.stockData.csrfBB[h]));
                      if (null != e && e[1]) {
                        d = e[1];
                        break;
                      }
                    }
                    if (d) {
                      a.csrf = d[1];
                      d = null;
                      for (h = 0; h < c.stockData.offerIdBB.length; h++) {
                        if (e = f.match(new RegExp(c.stockData.offerIdBB[h])), null != e && e[1]) {
                          d = e[1];
                          break;
                        }
                      }
                      d && (a.oid = d, a.callback());
                    }
                  } else {
                    b({error:"stock retrieval failed for offer: " + a.asin + " id: " + a.gid + " mismatch oid.", errorCode:10});
                  }
                }, !1, a.gid);
              })), a.callback = function() {
                return a.queue.shift()();
              }, c.stockSessions[a.domainId]) {
                a.cookies = c.stockSessions[a.domainId], a.callback();
              } else {
                var f = c.stockData.zipCodes[a.domainId];
                c.stockData.domainId == a.domainId ? (a.requestType = 3, p.httpPost("https://" + a.host + c.stockData.addressChangeUrl, c.stockData.addressChangePOST.replace("{ZIPCODE}", f), null, !1, a.gid)) : (a.requestType = 1, p.httpGet("https://" + a.host + c.stockData.geoUrl, function(d, e) {
                  d = d.match(new RegExp(c.stockData.csrfGeo));
                  if (null != d) {
                    a.csrf = d[1], a.requestType = 5, p.httpGet("https://" + a.host + c.stockData.toasterUrl.replace("{TIME_MS}", Date.now()), function(d) {
                      a.requestType = 2;
                      p.httpGet("https://" + a.host + c.stockData.setAddressUrl, function(d) {
                        d = d.match(new RegExp(c.stockData.csrfSetAddress));
                        null != d && (a.csrf = d[1]);
                        a.requestType = 3;
                        p.httpPost("https://" + a.host + c.stockData.addressChangeUrl, c.stockData.addressChangePOST.replace("{ZIPCODE}", f), null, !1, a.gid);
                      }, !1, a.gid);
                    }, !1, a.gid);
                  } else {
                    if (429 == e) {
                      var h = a.isMainRetry;
                      setTimeout(function() {
                        h ? b({error:"stock retrieval failed for offer: " + a.asin + " id: " + a.gid + " main.", errorCode:429}) : (a.isMainRetry = !0, c.addStockJob(a, b));
                      }, 1156);
                      h || (c.stockJobQueue.shift(), 0 < c.stockJobQueue.length && c.processStockJob(c.stockJobQueue[0][0], c.stockJobQueue[0][1]));
                    } else {
                      b({error:"stock retrieval failed for offer: " + a.asin + " id: " + a.gid + " main.", errorCode:7});
                    }
                  }
                }, !1, a.gid));
              }
            } else {
              console.log("could not init stock retrieval", c.stockInit, typeof chrome.webRequest), b({error:"could not init stock retrieval", errorCode:"undefined" != typeof chrome.webRequest ? 3 : 33});
            }
          }, 20));
        } else {
          console.log("stock retrieval not pro"), b({error:"stock retrieval failed, not subscribed", errorCode:2});
        }
      }
    }
  }, set:function(a, b, f) {
    var d = {};
    d[a] = b;
    c.storage.set(d, f);
  }, remove:function(a, b) {
    c.storage.remove(a, b);
  }, get:function(a, b) {
    "function" != typeof b && (b = function() {
    });
    c.storage.get(a, function(a) {
      b(a);
    });
  }};
  c.contextMenu();
  var p = {quiet:!0, version:chrome.runtime.getManifest().version, browser:1, url:"https://keepa.com", testUrl:"https://test.keepa.com", getDomain:function(a) {
    switch(a) {
      case "com":
        return 1;
      case "co.uk":
        return 2;
      case "de":
        return 3;
      case "fr":
        return 4;
      case "co.jp":
        return 5;
      case "ca":
        return 6;
      case "it":
        return 8;
      case "es":
        return 9;
      case "in":
        return 10;
      case "com.mx":
        return 11;
      case "com.br":
        return 12;
      case "com.au":
        return 13;
      case "nl":
        return 14;
      default:
        return 1;
    }
  }, objectStorage:[], Guid:function() {
    var a = function(c, d, b) {
      return c.length >= d ? c : a(b + c, d, b || " ");
    }, c = function() {
      var a = (new Date).getTime();
      return "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".replace(/x/g, function(c) {
        var d = (a + 16 * Math.random()) % 16 | 0;
        a = Math.floor(a / 16);
        return ("x" === c ? d : d & 7 | 8).toString(16);
      });
    };
    return {newGuid:function() {
      var b = "undefined" != typeof window.crypto.getRandomValues;
      if ("undefined" != typeof window.crypto && b) {
        b = new window.Uint16Array(16);
        window.crypto.getRandomValues(b);
        var d = "";
        for (h in b) {
          var e = b[h].toString(16);
          e = a(e, 4, "0");
          d += e;
        }
        var h = d;
      } else {
        h = c();
      }
      return h;
    }};
  }(), register:function() {
    chrome.cookies.onChanged.addListener(function(a) {
      a.removed || null == a.cookie || "keepa.com" != a.cookie.domain || "/extension" != a.cookie.path || ("token" == a.cookie.name ? C != a.cookie.value && 64 == a.cookie.value.length && (C = a.cookie.value, c.set("token", C), chrome.tabs.query({}, function(a) {
        try {
          a.forEach(function(a) {
            try {
              a.url && !a.incognito && chrome.tabs.sendMessage(a.id, {key:"updateToken", value:C});
            } catch (h) {
              console.log(h);
            }
          });
        } catch (w) {
          console.log(w);
        }
        document.location.reload(!1);
      })) : c.set(a.cookie.name, a.cookie.value));
    });
    var a = !1, b = function(b) {
      for (var d = {}, f = 0; f < b.length; d = {$jscomp$loop$prop$name$76:d.$jscomp$loop$prop$name$76}, f++) {
        d.$jscomp$loop$prop$name$76 = b[f];
        try {
          chrome.cookies.get({url:"https://keepa.com/extension", name:d.$jscomp$loop$prop$name$76}, function(b) {
            return function(d) {
              chrome.runtime.lastError && -1 < chrome.runtime.lastError.message.indexOf("No host permission") ? a || (a = !0, p.reportBug("extensionPermission restricted ### " + chrome.runtime.lastError.message)) : null != d && null != d.value && 0 < d.value.length && c.set(b.$jscomp$loop$prop$name$76, d.value);
            };
          }(d));
        } catch (h) {
          console.log(h);
        }
      }
    };
    b(B);
    chrome.cookies.get({url:"https://keepa.com/extension", name:"token"}, function(a) {
      if (null != a && 64 == a.value.length) {
        C = a.value, c.set("token", C);
      } else {
        var d = (Date.now() / 1000 | 0) + 31536E3;
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"optOut_crawl", value:"0", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"revealStock", value:"1", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"s_boxType", value:"0", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"s_boxOfferListing", value:"1", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"s_boxHorizontal", value:"0", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"webGraphType", value:"[1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"webGraphRange", value:"180", secure:!0, expirationDate:d});
        chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"overlayPriceGraph", value:"0", secure:!0, expirationDate:d});
        b(B);
        c.get("token", function(a) {
          C = (a = a.token) && 64 == a.length ? a : p.Guid.newGuid();
          chrome.cookies.set({url:"https://keepa.com", path:"/extension", name:"token", value:C, secure:!0, expirationDate:d});
        });
      }
    });
    try {
      "undefined" != typeof chrome.storage.sync && chrome.storage.sync.clear();
    } catch (f) {
    }
    window.addEventListener("message", function(a) {
      var d = a.data;
      if (d) {
        if ("string" === typeof d) {
          try {
            d = JSON.parse(d);
          } catch (Z) {
            return;
          }
        }
        if (d.log) {
          console.log(d.log);
        } else {
          var c = function() {
          };
          if (a.origin != p.url && a.origin != p.testUrl) {
            var b = u.getMessage();
            if (null != b && ("function" == typeof b.onDoneC && (c = b.onDoneC, delete b.onDoneC), "undefined" == typeof b.sent && d.sandbox && a.source == document.getElementById("keepa_data").contentWindow)) {
              if (d.sandbox == b.url) {
                u.setStatTime(40);
                try {
                  a.source.postMessage({key:"data", value:b}, "*");
                } catch (Z) {
                  u.abortJob(407), c();
                }
              } else {
                d.isUrlMsg ? (b.wasUrl = d.sandbox, u.abortJob(405)) : (a = u.getOutgoingMessage(b, d.sandbox), m.sendMessage(a)), c();
              }
            }
          }
        }
      }
    });
    A ? c.set("addonVersionFirefox", p.version) : c.set("addonVersionChrome", p.version);
    try {
      chrome.runtime.setUninstallURL("https://dyn.keepa.com/app/stats/?type=uninstall&version=" + Q + "." + p.version);
    } catch (f) {
    }
    window.setTimeout(function() {
      m.initWebSocket();
    }, 2000);
  }, log:function(a) {
    c.log(a);
  }, lastBugReport:0, reportBug:function(a, b, f) {
    var d = Error();
    c.get(["token"], function(c) {
      var e = Date.now();
      if (!(12E5 > e - p.lastBugReport || /(dead object)|(Script error)|(setUninstallURL)|(File error: Corrupted)|(operation is insecure)|(\.location is null)/i.test(a))) {
        p.lastBugReport = e;
        e = "";
        var r = p.version;
        b = b || "";
        try {
          if (e = d.stack.split("\n").splice(1).splice(1).join("&ensp;&lArr;&ensp;"), !/(keepa|content)\.js/.test(e) || e.startsWith("https://www.amazon") || e.startsWith("https://smile.amazon") || e.startsWith("https://sellercentral")) {
            return;
          }
        } catch (S) {
        }
        try {
          e = e.replace(/chrome-extension:\/\/.*?\/content\//g, "").replace(/:[0-9]*?\)/g, ")").replace(/[ ]{2,}/g, "");
        } catch (S) {
        }
        if ("object" == typeof a) {
          try {
            a = a instanceof Error ? a.toString() : JSON.stringify(a);
          } catch (S) {
          }
        }
        null == f && (f = {exception:a, additional:b, url:document.location.host, stack:e});
        f.keepaType = Q;
        f.version = r;
        setTimeout(function() {
          p.httpPost("https://dyn.keepa.com/service/bugreport/?user=" + c.token + "&type=" + ca + "&version=" + r, JSON.stringify(f), null, !1);
        }, 50);
      }
    });
  }, httpGet:function(a, b, c, d) {
    var e = new XMLHttpRequest;
    b && (e.onreadystatechange = function() {
      4 == e.readyState && b.call(this, e.responseText, e.status);
    });
    e.withCredentials = c;
    e.open("GET", a, !0);
    d && e.setRequestHeader("krequestid", d);
    e.send();
  }, httpPost:function(a, b, c, d, e) {
    var f = new XMLHttpRequest;
    c && (f.onreadystatechange = function() {
      4 == f.readyState && c.call(this, f.responseText, f.status);
    });
    f.withCredentials = d;
    f.open("POST", a, !0);
    f.setRequestHeader("Content-Type", "text/plain;charset=UTF-8");
    e && f.setRequestHeader("krequestid", e);
    f.send(b);
  }};
  window.addEventListener("error", function(a, b, c, d, e) {
    a = "object" === typeof a && a.srcElement && a.target ? "[object HTMLScriptElement]" == a.srcElement && "[object HTMLScriptElement]" == a.target ? "Error loading script " + JSON.stringify(a) : JSON.stringify(a) : a.toString();
    var f = "";
    d = d || 0;
    if (e && e.stack) {
      f = e.stack;
      try {
        f = e.stack.split("\n").splice(1).splice(1).join("&ensp;&lArr;&ensp;");
        if (!/(keepa|content)\.js/.test(f)) {
          return;
        }
        f = f.replace(/chrome-extension:\/\/.*?\/content\//g, "").replace(/:[0-9]*?\)/g, ")").replace(/[ ]{2,}/g, "");
      } catch (Z) {
      }
    }
    a = {msg:a, url:(b || document.location.toString()) + ":" + parseInt(c || 0) + ":" + parseInt(d || 0), stack:f};
    "ipbakfmnjdenbmoenhicfmoojdojjjem" != chrome.runtime.id && "blfpbjkajgamcehdbehfdioapoiibdmc" != chrome.runtime.id || console.log(a);
    p.reportBug(null, null, a);
    return !1;
  });
  var aa = 0;
  var m = {server:["wss://dyn.keepa.com", "wss://dyn-2.keepa.com"], serverIndex:0, clearTimeout:0, webSocket:null, sendPlainMessage:function(a) {
    K || (a = JSON.stringify(a), m.webSocket.send(pako.deflate(a)));
  }, sendMessage:function(a) {
    if (!K) {
      u.clearIframe();
      var c = pako.deflate(JSON.stringify(a));
      u.clearMessage();
      1 == m.webSocket.readyState && m.webSocket.send(c);
      403 == a.status && u.endSession(aa);
      b.console.clear();
    }
  }, initWebSocket:function() {
    K || c.get(["token", "optOut_crawl"], function(a) {
      var b = a.token, f = a.optOut_crawl;
      if (b && 64 == b.length) {
        var d = function() {
          if (null == m.webSocket || 1 != m.webSocket.readyState) {
            m.serverIndex %= m.server.length;
            if ("undefined" == typeof f || "undefined" == f || null == f || "null" == f || "NaN" == f) {
              f = "0";
            }
            g && (f = "1");
            "undefined" === typeof chrome.webRequest && (f = "1");
            var a = new WebSocket(m.server[m.serverIndex] + "/apps/cloud/?app=" + Q + "&version=" + p.version + "&i=" + J + "&wr=" + typeof chrome.webRequest + "&optOut=" + f, b);
            a.binaryType = "arraybuffer";
            a.onmessage = function(a) {
              a = a.data;
              var b = null;
              a instanceof ArrayBuffer && (a = pako.inflate(a, {to:"string"}));
              try {
                b = JSON.parse(a);
              } catch (S) {
                p.reportBug(S, a);
                return;
              }
              108 == b.status ? 1 === b.guest ? (c.isGuest = !0, c.actionUrl = b.actionUrl) : c.isGuest = !1 : "" == b.key ? c.stockData.domainId = b.domainId : 108108 == b.timeout ? (b.stockData && (c.stockData = b.stockData, console.log("stock reveal ready"), b.stockData.amazonIds && (e = b.stockData.amazonIds)), "undefined" != typeof b.keepaBoxPlaceholder && c.set("keepaBoxPlaceholder", b.keepaBoxPlaceholder), "undefined" != typeof b.keepaBoxPlaceholderBackup && c.set("keepaBoxPlaceholderBackup", 
              b.keepaBoxPlaceholderBackup), "undefined" != typeof b.keepaBoxPlaceholderBackupClass && c.set("keepaBoxPlaceholderBackupClass", b.keepaBoxPlaceholderBackupClass), "undefined" != typeof b.keepaBoxPlaceholderAppend && c.set("keepaBoxPlaceholderAppend", b.keepaBoxPlaceholderAppend), "undefined" != typeof b.keepaBoxPlaceholderBackupAppend && c.set("keepaBoxPlaceholderBackupAppend", b.keepaBoxPlaceholderBackupAppend)) : (b.domainId && (aa = b.domainId), u.clearIframe(), u.onMessage(b));
            };
            a.onclose = function(a) {
              setTimeout(function() {
                d();
              }, 36E4 * Math.random());
            };
            a.onerror = function(b) {
              m.serverIndex++;
              a.close();
            };
            a.onopen = function() {
              u.abortJob(414);
            };
            m.webSocket = a;
          }
        };
        d();
      }
    });
  }};
  var u = function() {
    function a(a) {
      try {
        n.stats.times.push(a), n.stats.times.push(Date.now() - n.stats.start);
      } catch (v) {
      }
    }
    function g(b, c) {
      b.sent = !0;
      a(25);
      var d = b.key, t = b.messageId;
      b = b.stats;
      try {
        var e = D[E]["session-id"];
      } catch (k) {
        e = "";
      }
      d = {key:d, messageId:t, stats:b, sessionId:e, payload:[], status:200};
      for (var f in c) {
        d[f] = c[f];
      }
      return d;
    }
    function f(b) {
      E = n.domainId;
      V = B(D);
      "object" != typeof D[E] && (D[E] = {});
      "undefined" == typeof n.headers.Accept && (n.headers.Accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*!/!*;q=0.8");
      l(b, !b.isAjax, function(c) {
        a(0);
        var d = {payload:[]};
        if (c.match(K)) {
          d.status = 403;
        } else {
          if (b.contentFilters && 0 < b.contentFilters.length) {
            for (var e in b.contentFilters) {
              var t = c.match(new RegExp(b.contentFilters[e]));
              if (t) {
                d.payload[e] = t[1].replace(/\n/g, "");
              } else {
                d.status = 305;
                d.payload[e] = c;
                break;
              }
            }
          } else {
            d.payload = [c];
          }
        }
        try {
          b.stats.times.push(3), b.stats.times.push(p.lastBugReport);
        } catch (z) {
        }
        "undefined" == typeof b.sent && (d = g(b, d), m.sendMessage(d));
      });
    }
    function d(b) {
      E = n.domainId;
      V = B(D);
      "object" != typeof D[E] && (D[E] = {});
      "undefined" == typeof n.headers.Accept && (n.headers.Accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*!/!*;q=0.8");
      a(4);
      var d = new URL(b.url), f = null;
      try {
        null != b.scrapeFilters && 0 < b.scrapeFilters.length && b.scrapeFilters[0].lager && chrome.cookies.get({url:d.origin, name:"session-id"}, function(a) {
          null == a ? f = "guest" : null != a.value && 5 < a.value.length && (f = a.value);
        });
      } catch (x) {
      }
      l(b, !b.isAjax, function(v, t) {
        a(6);
        if ("undefined" == typeof b.sent) {
          var z = {};
          try {
            for (var k = v.evaluate("//comment()", v, null, XPathResult.ANY_TYPE, null), h = k.iterateNext(), n = ""; h;) {
              n += h.textContent, h = k.iterateNext();
            }
            if (v.querySelector("body").textContent.match(K) || n.match(K)) {
              z.status = 403;
              if ("undefined" != typeof b.sent) {
                return;
              }
              z = g(b, z);
              m.sendMessage(z);
              return;
            }
          } catch (H) {
          }
          a(7);
          if (b.scrapeFilters && 0 < b.scrapeFilters.length) {
            var T = {}, x = {}, r = {}, l = "", I = null, u = function() {
              if ("" === l) {
                z.payload = [I];
                z.scrapedData = r;
                for (var a in x) {
                  z[a] = x[a];
                }
              } else {
                z.status = 305, z.payload = [I, l, ""];
              }
              try {
                b.stats.times.push(99), b.stats.times.push(p.lastBugReport);
              } catch (fa) {
              }
              "undefined" == typeof b.sent && (z = g(b, z), m.sendMessage(z));
            }, A = function(a, b, c) {
              var d = [];
              if (!a.selector) {
                if (!a.regExp) {
                  return l = "invalid selector, sel/regexp", !1;
                }
                d = v.querySelector("html").innerHTML.match(new RegExp(a.regExp));
                if (!d || d.length < a.reGroup) {
                  c = "regexp fail: html - " + a.name + c;
                  if (!1 === a.optional) {
                    return l = c, !1;
                  }
                  I += " // " + c;
                  return !0;
                }
                return d[a.reGroup];
              }
              var e = b.querySelectorAll(a.selector);
              0 == e.length && (e = b.querySelectorAll(a.altSelector));
              if (0 == e.length) {
                if (!0 === a.optional) {
                  return !0;
                }
                l = "selector no match: " + a.name + c;
                return !1;
              }
              if (a.parentSelector && (e = [e[0].parentNode.querySelector(a.parentSelector)], null == e[0])) {
                if (!0 === a.optional) {
                  return !0;
                }
                l = "parent selector no match: " + a.name + c;
                return !1;
              }
              if ("undefined" != typeof a.multiple && null != a.multiple && (!0 === a.multiple && 1 > e.length || !1 === a.multiple && 1 < e.length)) {
                c = "selector multiple mismatch: " + a.name + c + " found: " + e.length;
                if (!1 === a.optional) {
                  return l = c, !1;
                }
                I += " // " + c;
                return !0;
              }
              if (a.isListSelector) {
                return T[a.name] = e, !0;
              }
              if (!a.attribute) {
                return l = "selector attribute undefined?: " + a.name + c, !1;
              }
              for (var f in e) {
                if (e.hasOwnProperty(f)) {
                  b = e[f];
                  if (!b) {
                    break;
                  }
                  if (a.childNode) {
                    a.childNode = Number(a.childNode);
                    b = b.childNodes;
                    if (b.length < a.childNode) {
                      c = "childNodes fail: " + b.length + " - " + a.name + c;
                      if (!1 === a.optional) {
                        return l = c, !1;
                      }
                      I += " // " + c;
                      return !0;
                    }
                    b = b[a.childNode];
                  }
                  b = "text" == a.attribute ? b.textContent : "html" == a.attribute ? b.innerHTML : b.getAttribute(a.attribute);
                  if (!b || 0 == b.length || 0 == b.replace(/(\r\n|\n|\r)/gm, "").replace(/^\s+|\s+$/g, "").length) {
                    c = "selector attribute null: " + a.name + c;
                    if (!1 === a.optional) {
                      return l = c, !1;
                    }
                    I += " // " + c;
                    return !0;
                  }
                  if (a.regExp) {
                    var t = b.match(new RegExp(a.regExp));
                    if (!t || t.length < a.reGroup) {
                      c = "regexp fail: " + b + " - " + a.name + c;
                      if (!1 === a.optional) {
                        return l = c, !1;
                      }
                      I += " // " + c;
                      return !0;
                    }
                    d.push("undefined" == typeof t[a.reGroup] ? t[0] : t[a.reGroup]);
                  } else {
                    d.push(b);
                  }
                  if (!a.multiple) {
                    break;
                  }
                }
              }
              return a.multiple ? d : d[0];
            };
            h = !1;
            k = {};
            for (var B in b.scrapeFilters) {
              k.$jscomp$loop$prop$pageType$81 = B;
              a: {
                if (h) {
                  break;
                }
                k.$jscomp$loop$prop$pageFilter$78 = b.scrapeFilters[k.$jscomp$loop$prop$pageType$81];
                k.$jscomp$loop$prop$pageVersionTest$79 = k.$jscomp$loop$prop$pageFilter$78.pageVersionTest;
                n = v.querySelectorAll(k.$jscomp$loop$prop$pageVersionTest$79.selector);
                0 == n.length && (n = v.querySelectorAll(k.$jscomp$loop$prop$pageVersionTest$79.altSelector));
                if (0 != n.length) {
                  if ("undefined" != typeof k.$jscomp$loop$prop$pageVersionTest$79.multiple && null != k.$jscomp$loop$prop$pageVersionTest$79.multiple) {
                    if (!0 === k.$jscomp$loop$prop$pageVersionTest$79.multiple && 2 > n.length) {
                      break a;
                    }
                    if (!1 === k.$jscomp$loop$prop$pageVersionTest$79.multiple && 1 < n.length) {
                      break a;
                    }
                  }
                  if (k.$jscomp$loop$prop$pageVersionTest$79.attribute) {
                    var w = null;
                    w = "text" == k.$jscomp$loop$prop$pageVersionTest$79.attribute ? "" : n[0].getAttribute(k.$jscomp$loop$prop$pageVersionTest$79.attribute);
                    if (null == w) {
                      break a;
                    }
                  }
                  var D = k.$jscomp$loop$prop$pageType$81;
                  k.$jscomp$loop$prop$revealMAP$99 = k.$jscomp$loop$prop$pageFilter$78.revealMAP;
                  k.$jscomp$loop$prop$revealed$101 = !1;
                  k.$jscomp$loop$prop$afterAjaxFinished$102 = function(t) {
                    return function() {
                      var k = 0, h = [];
                      a(26);
                      var g = {}, n;
                      for (n in t.$jscomp$loop$prop$pageFilter$78) {
                        g.$jscomp$loop$prop$sel$88 = t.$jscomp$loop$prop$pageFilter$78[n];
                        if (!(g.$jscomp$loop$prop$sel$88.name == t.$jscomp$loop$prop$pageVersionTest$79.name || t.$jscomp$loop$prop$revealed$101 && "revealMAP" == g.$jscomp$loop$prop$sel$88.name)) {
                          var p = v;
                          if (g.$jscomp$loop$prop$sel$88.parentList) {
                            var l = [];
                            if ("undefined" != typeof T[g.$jscomp$loop$prop$sel$88.parentList]) {
                              l = T[g.$jscomp$loop$prop$sel$88.parentList];
                            } else {
                              if (!0 === A(t.$jscomp$loop$prop$pageFilter$78[g.$jscomp$loop$prop$sel$88.parentList], p, t.$jscomp$loop$prop$pageType$81)) {
                                l = T[g.$jscomp$loop$prop$sel$88.parentList];
                              } else {
                                break;
                              }
                            }
                            x[g.$jscomp$loop$prop$sel$88.parentList] || (x[g.$jscomp$loop$prop$sel$88.parentList] = []);
                            p = 0;
                            var q = {}, m;
                            for (m in l) {
                              if (l.hasOwnProperty(m)) {
                                if ("lager" == g.$jscomp$loop$prop$sel$88.name) {
                                  p++;
                                  try {
                                    q.$jscomp$loop$prop$sellerId$86 = null;
                                    var H = t.$jscomp$loop$prop$pageFilter$78.sellerId;
                                    q.$jscomp$loop$prop$currentASIN$83 = b.url.match(/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/)[1];
                                    if (!(null == q.$jscomp$loop$prop$currentASIN$83 || 9 > q.$jscomp$loop$prop$currentASIN$83.length || "undefined" == typeof H || null == H || null == q.$jscomp$loop$prop$currentASIN$83 || 2 > q.$jscomp$loop$prop$currentASIN$83.length)) {
                                      q.$jscomp$loop$prop$sellerId$86 = x[H.parentList][m] && x[H.parentList][m][H.name] ? x[H.parentList][m][H.name] : A(H, l[m], t.$jscomp$loop$prop$pageType$81);
                                      var y = !1;
                                      try {
                                        x[H.parentList][m] && x[H.parentList][m].sellerName && -1 < x[H.parentList][m].sellerName.indexOf("Amazon") && (null == q.$jscomp$loop$prop$sellerId$86 || 12 > (q.$jscomp$loop$prop$sellerId$86 + "").length) && (y = !0);
                                      } catch (O) {
                                        console.error(O);
                                      }
                                      q.$jscomp$loop$prop$sellerId$86 = y ? e[b.domainId] : q.$jscomp$loop$prop$sellerId$86.match(/&seller=([A-Z0-9]{9,21})($|&)/)[1];
                                      y = void 0;
                                      q.$jscomp$loop$prop$offerId$84 = void 0;
                                      g.$jscomp$loop$prop$sel$88.selector && (y = l[m].querySelector(g.$jscomp$loop$prop$sel$88.selector));
                                      g.$jscomp$loop$prop$sel$88.altSelector && (q.$jscomp$loop$prop$offerId$84 = l[m].querySelector(g.$jscomp$loop$prop$sel$88.altSelector));
                                      q.$jscomp$loop$prop$offerId$84 && (q.$jscomp$loop$prop$offerId$84 = q.$jscomp$loop$prop$offerId$84.getAttribute(g.$jscomp$loop$prop$sel$88.attribute));
                                      q.$jscomp$loop$prop$maxQty$85 = 999;
                                      if (!q.$jscomp$loop$prop$offerId$84) {
                                        try {
                                          var w = JSON.parse(g.$jscomp$loop$prop$sel$88.regExp);
                                          if (w.sel1) {
                                            try {
                                              var B = JSON.parse(l[m].querySelectorAll(w.sel1)[0].dataset[w.dataSet1]);
                                              q.$jscomp$loop$prop$offerId$84 = B[w.val1];
                                              q.$jscomp$loop$prop$maxQty$85 = B.maxQty;
                                            } catch (O) {
                                            }
                                          }
                                          if (!q.$jscomp$loop$prop$offerId$84 && w.sel2) {
                                            try {
                                              var D = JSON.parse(l[m].querySelectorAll(w.sel2)[0].dataset[w.dataSet2]);
                                              q.$jscomp$loop$prop$offerId$84 = D[w.val2];
                                              q.$jscomp$loop$prop$maxQty$85 = D.maxQty;
                                            } catch (O) {
                                            }
                                          }
                                        } catch (O) {
                                        }
                                      }
                                      if (y && q.$jscomp$loop$prop$sellerId$86 && null != f) {
                                        k++;
                                        q.$jscomp$loop$prop$mapIndex$91 = m + "";
                                        q.$jscomp$loop$prop$isMAP$89 = !1;
                                        try {
                                          q.$jscomp$loop$prop$isMAP$89 = x[g.$jscomp$loop$prop$sel$88.parentList][q.$jscomp$loop$prop$mapIndex$91].isMAP || -1 != l[m].textContent.toLowerCase().indexOf("add to cart to see product details");
                                        } catch (O) {
                                        }
                                        q.$jscomp$loop$prop$busy$90 = !0;
                                        setTimeout(function(a, e) {
                                          return function() {
                                            c.addStockJob({type:"getStock", asin:a.$jscomp$loop$prop$currentASIN$83, oid:a.$jscomp$loop$prop$offerId$84, host:d.host, maxQty:a.$jscomp$loop$prop$maxQty$85, sellerId:a.$jscomp$loop$prop$sellerId$86, onlyMaxQty:9 == e.$jscomp$loop$prop$sel$88.reGroup, isMAP:a.$jscomp$loop$prop$isMAP$89, referer:d.host + "/dp/" + a.$jscomp$loop$prop$currentASIN$83, domainId:b.domainId, force:!0, session:f}, function(t) {
                                              a.$jscomp$loop$prop$busy$90 && ("undefined" == typeof t || null != t.error && 430 == t.errorCode ? c.addStockJob({type:"getStock", asin:a.$jscomp$loop$prop$currentASIN$83, oid:a.$jscomp$loop$prop$offerId$84, host:d.host, maxQty:a.$jscomp$loop$prop$maxQty$85, sellerId:a.$jscomp$loop$prop$sellerId$86, onlyMaxQty:9 == e.$jscomp$loop$prop$sel$88.reGroup, isMAP:a.$jscomp$loop$prop$isMAP$89, referer:d.host + "/dp/" + a.$jscomp$loop$prop$currentASIN$83, domainId:b.domainId, 
                                              force:!0, session:f}, function(b) {
                                                a.$jscomp$loop$prop$busy$90 && (a.$jscomp$loop$prop$busy$90 = !1, "undefined" != typeof b && (x[e.$jscomp$loop$prop$sel$88.parentList][a.$jscomp$loop$prop$mapIndex$91][e.$jscomp$loop$prop$sel$88.name] = b), 0 == --k && u(z));
                                              }) : (a.$jscomp$loop$prop$busy$90 = !1, x[e.$jscomp$loop$prop$sel$88.parentList][a.$jscomp$loop$prop$mapIndex$91][e.$jscomp$loop$prop$sel$88.name] = t, 0 == --k && u(z)));
                                            });
                                            setTimeout(function() {
                                              a.$jscomp$loop$prop$busy$90 && 0 == --k && (a.$jscomp$loop$prop$busy$90 = !1, console.log("timeout " + a.$jscomp$loop$prop$offerId$84), u(z));
                                            }, 3000 + 800 * k);
                                          };
                                        }(q, g), 1);
                                      }
                                    }
                                  } catch (O) {
                                  }
                                } else {
                                  if ("revealMAP" == g.$jscomp$loop$prop$sel$88.name) {
                                    if (q.$jscomp$loop$prop$revealMAP$54$92 = g.$jscomp$loop$prop$sel$88, y = void 0, y = q.$jscomp$loop$prop$revealMAP$54$92.selector ? l[m].querySelector(q.$jscomp$loop$prop$revealMAP$54$92.selector) : l[m], null != y && y.textContent.match(new RegExp(q.$jscomp$loop$prop$revealMAP$54$92.regExp))) {
                                      y = b.url.match(/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/)[1];
                                      var E = t.$jscomp$loop$prop$pageFilter$78.sellerId;
                                      "undefined" == typeof E || null == E || null == y || 2 > y.length || (E = l[m].querySelector(g.$jscomp$loop$prop$sel$88.childNode).value, null == E || 20 > E + 0 || (y = q.$jscomp$loop$prop$revealMAP$54$92.altSelector.replace("OFFERID", E).replace("ASINID", y), k++, q.$jscomp$loop$prop$mapIndex$58$93 = m + "", C(y, "GET", null, 3000, function(a) {
                                        return function(b) {
                                          try {
                                            var c = t.$jscomp$loop$prop$pageFilter$78.price;
                                            if (c && c.regExp) {
                                              if (b.match(/no valid offer--/)) {
                                                x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93] || (x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93] = {}), x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93][a.$jscomp$loop$prop$revealMAP$54$92.name] = -1;
                                              } else {
                                                var d = b.match(new RegExp("price info--\x3e(?:.|\\n)*?" + c.regExp + "(?:.|\\n)*?\x3c!--")), e = b.match(/price info--\x3e(?:.|\n)*?(?:<span.*?size-small.*?">)([^]*?<\/span)(?:.|\n)*?\x3c!--/);
                                                if (!d || d.length < c.reGroup) {
                                                  I += " //  priceMAP regexp fail: " + (b + " - " + c.name + t.$jscomp$loop$prop$pageType$81);
                                                } else {
                                                  var f = d[c.reGroup];
                                                  x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93] || (x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93] = {});
                                                  x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93][a.$jscomp$loop$prop$revealMAP$54$92.name] = f;
                                                  null != e && 2 == e.length && (x[a.$jscomp$loop$prop$revealMAP$54$92.parentList][a.$jscomp$loop$prop$mapIndex$58$93][a.$jscomp$loop$prop$revealMAP$54$92.name + "Shipping"] = e[1].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                                                }
                                              }
                                            }
                                          } catch (ha) {
                                          }
                                          0 == --k && 0 == h.length && u();
                                        };
                                      }(q), function() {
                                        0 == --k && 0 == h.length && u();
                                      })));
                                    }
                                  } else {
                                    y = A(g.$jscomp$loop$prop$sel$88, l[m], t.$jscomp$loop$prop$pageType$81);
                                    if (!1 === y) {
                                      break;
                                    }
                                    if (!0 !== y) {
                                      if (x[g.$jscomp$loop$prop$sel$88.parentList][m] || (x[g.$jscomp$loop$prop$sel$88.parentList][m] = {}), g.$jscomp$loop$prop$sel$88.multiple) {
                                        for (var F in y) {
                                          y.hasOwnProperty(F) && !g.$jscomp$loop$prop$sel$88.keepBR && (y[F] = y[F].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                                        }
                                        y = y.join("\u271c\u271c");
                                        x[g.$jscomp$loop$prop$sel$88.parentList][m][g.$jscomp$loop$prop$sel$88.name] = y;
                                      } else {
                                        x[g.$jscomp$loop$prop$sel$88.parentList][m][g.$jscomp$loop$prop$sel$88.name] = g.$jscomp$loop$prop$sel$88.keepBR ? y : y.replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " ");
                                      }
                                    }
                                  }
                                }
                              }
                              q = {$jscomp$loop$prop$currentASIN$83:q.$jscomp$loop$prop$currentASIN$83, $jscomp$loop$prop$offerId$84:q.$jscomp$loop$prop$offerId$84, $jscomp$loop$prop$maxQty$85:q.$jscomp$loop$prop$maxQty$85, $jscomp$loop$prop$sellerId$86:q.$jscomp$loop$prop$sellerId$86, $jscomp$loop$prop$isMAP$89:q.$jscomp$loop$prop$isMAP$89, $jscomp$loop$prop$busy$90:q.$jscomp$loop$prop$busy$90, $jscomp$loop$prop$mapIndex$91:q.$jscomp$loop$prop$mapIndex$91, $jscomp$loop$prop$revealMAP$54$92:q.$jscomp$loop$prop$revealMAP$54$92, 
                              $jscomp$loop$prop$mapIndex$58$93:q.$jscomp$loop$prop$mapIndex$58$93};
                            }
                          } else {
                            l = A(g.$jscomp$loop$prop$sel$88, p, t.$jscomp$loop$prop$pageType$81);
                            if (!1 === l) {
                              break;
                            }
                            if (!0 !== l) {
                              if (g.$jscomp$loop$prop$sel$88.multiple) {
                                for (var G in l) {
                                  l.hasOwnProperty(G) && !g.$jscomp$loop$prop$sel$88.keepBR && (l[G] = l[G].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                                }
                                l = l.join();
                              } else {
                                g.$jscomp$loop$prop$sel$88.keepBR || (l = l.replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                              }
                              r[g.$jscomp$loop$prop$sel$88.name] = l;
                            }
                          }
                        }
                        g = {$jscomp$loop$prop$sel$88:g.$jscomp$loop$prop$sel$88};
                      }
                      try {
                        if (1 == h.length || "500".endsWith("8") && 0 < h.length) {
                          h.shift()();
                        } else {
                          for (g = 0; g < h.length; g++) {
                            setTimeout(function() {
                              0 < h.length && h.shift()();
                            }, 500 * g);
                          }
                        }
                      } catch (O) {
                      }
                      0 == k && 0 == h.length && u();
                    };
                  }(k);
                  if (k.$jscomp$loop$prop$revealMAP$99) {
                    if (h = v.querySelector(k.$jscomp$loop$prop$revealMAP$99.selector), null != h) {
                      k.$jscomp$loop$prop$url$100 = h.getAttribute(k.$jscomp$loop$prop$revealMAP$99.attribute);
                      if (null == k.$jscomp$loop$prop$url$100 || 0 == k.$jscomp$loop$prop$url$100.length) {
                        k.$jscomp$loop$prop$afterAjaxFinished$102();
                        break;
                      }
                      0 != k.$jscomp$loop$prop$url$100.indexOf("http") && (h = document.createElement("a"), h.href = b.url, k.$jscomp$loop$prop$url$100 = h.origin + k.$jscomp$loop$prop$url$100);
                      r[k.$jscomp$loop$prop$revealMAP$99.name] = "1";
                      k.$jscomp$loop$prop$url$100 = k.$jscomp$loop$prop$url$100.replace(/(mapPopover.*?)(false)/, "$1true");
                      k.$jscomp$loop$prop$xhr$97 = new XMLHttpRequest;
                      k.$jscomp$loop$prop$hasTimeout$96 = !1;
                      k.$jscomp$loop$prop$ti$98 = setTimeout(function(a) {
                        return function() {
                          a.$jscomp$loop$prop$hasTimeout$96 = !0;
                          a.$jscomp$loop$prop$afterAjaxFinished$102();
                        };
                      }(k), 4000);
                      k.$jscomp$loop$prop$xhr$97.onreadystatechange = function(a) {
                        return function() {
                          if (!a.$jscomp$loop$prop$hasTimeout$96 && 4 == a.$jscomp$loop$prop$xhr$97.readyState) {
                            clearTimeout(a.$jscomp$loop$prop$ti$98);
                            if (200 == a.$jscomp$loop$prop$xhr$97.status) {
                              var b = a.$jscomp$loop$prop$xhr$97.responseText;
                              if (a.$jscomp$loop$prop$revealMAP$99.regExp) {
                                var c = b.match(new RegExp(a.$jscomp$loop$prop$revealMAP$99.regExp));
                                if (!c || c.length < a.$jscomp$loop$prop$revealMAP$99.reGroup) {
                                  if (c = v.querySelector(a.$jscomp$loop$prop$revealMAP$99.selector)) {
                                    var d = c.cloneNode(!1);
                                    d.innerHTML = b;
                                    c.parentNode.replaceChild(d, c);
                                  }
                                } else {
                                  r[a.$jscomp$loop$prop$revealMAP$99.name] = c[a.$jscomp$loop$prop$revealMAP$99.reGroup], r[a.$jscomp$loop$prop$revealMAP$99.name + "url"] = a.$jscomp$loop$prop$url$100;
                                }
                              }
                            }
                            a.$jscomp$loop$prop$revealed$101 = !0;
                            a.$jscomp$loop$prop$afterAjaxFinished$102();
                          }
                        };
                      }(k);
                      k.$jscomp$loop$prop$xhr$97.onerror = k.$jscomp$loop$prop$afterAjaxFinished$102;
                      k.$jscomp$loop$prop$xhr$97.open("GET", k.$jscomp$loop$prop$url$100, !0);
                      k.$jscomp$loop$prop$xhr$97.send();
                    } else {
                      k.$jscomp$loop$prop$afterAjaxFinished$102();
                    }
                  } else {
                    k.$jscomp$loop$prop$afterAjaxFinished$102();
                  }
                  h = !0;
                }
              }
              k = {$jscomp$loop$prop$pageFilter$78:k.$jscomp$loop$prop$pageFilter$78, $jscomp$loop$prop$pageVersionTest$79:k.$jscomp$loop$prop$pageVersionTest$79, $jscomp$loop$prop$revealed$101:k.$jscomp$loop$prop$revealed$101, $jscomp$loop$prop$pageType$81:k.$jscomp$loop$prop$pageType$81, $jscomp$loop$prop$hasTimeout$96:k.$jscomp$loop$prop$hasTimeout$96, $jscomp$loop$prop$afterAjaxFinished$102:k.$jscomp$loop$prop$afterAjaxFinished$102, $jscomp$loop$prop$xhr$97:k.$jscomp$loop$prop$xhr$97, $jscomp$loop$prop$ti$98:k.$jscomp$loop$prop$ti$98, 
              $jscomp$loop$prop$revealMAP$99:k.$jscomp$loop$prop$revealMAP$99, $jscomp$loop$prop$url$100:k.$jscomp$loop$prop$url$100};
            }
            a(8);
            if (null == D) {
              l += " // no pageVersion matched";
              z.payload = [I, l, b.dbg1 ? t : ""];
              z.status = 308;
              a(10);
              try {
                b.stats.times.push(99), b.stats.times.push(p.lastBugReport);
              } catch (H) {
              }
              "undefined" == typeof b.sent && (z = g(b, z), m.sendMessage(z));
            }
          } else {
            a(9), z.status = 306, "undefined" == typeof b.sent && (z = g(b, z), m.sendMessage(z));
          }
        }
      });
    }
    function l(c, d, e) {
      null == N || X || R();
      L = c;
      var f = c.messageId;
      setTimeout(function() {
        null != L && L.messageId == f && (L = L = null);
      }, c.timeout);
      c.onDoneC = function() {
        L = null;
      };
      if (d) {
        a(11), d = document.getElementById("keepa_data"), d.removeAttribute("srcdoc"), d.src = c.url;
      } else {
        var v = function(d) {
          a(12);
          if ("o0" == c.key) {
            e(d);
          } else {
            var f = document.getElementById("keepa_data_2");
            f.src = "";
            d = d.replace(/src=".*?"/g, 'src=""');
            if (null != c) {
              c.block && (d = d.replace(new RegExp(c.block, "g"), ""));
              a(13);
              var v = !1;
              f.srcdoc = d;
              a(18);
              f.onload = function() {
                a(19);
                v || (f.onload = void 0, v = !0, a(20), setTimeout(function() {
                  a(21);
                  var b = document.getElementById("keepa_data_2").contentWindow;
                  try {
                    e(b.document, d);
                  } catch (ba) {
                    p.reportBug(ba), G(410);
                  }
                }, 80));
              };
            }
            b.console.clear();
          }
        };
        d = 0;
        1 == c.httpMethod && (c.scrapeFilters && 0 < c.scrapeFilters.length && (J = c), Q || (Q = !0, c.l && 0 < c.l.length && (N = c.l, R(), d = 25)));
        setTimeout(function() {
          C(c.url, U[c.httpMethod], c.postData, c.timeout, v);
        }, d);
      }
    }
    function h() {
      try {
        var a = document.getElementById("keepa_data");
        a.src = "";
        a.removeAttribute("srcdoc");
      } catch (T) {
      }
      try {
        var b = document.getElementById("keepa_data_2");
        b.src = "";
        b.removeAttribute("srcdoc");
      } catch (T) {
      }
      L = null;
    }
    function C(b, c, d, e, f) {
      var v = new XMLHttpRequest;
      if (f) {
        var g = !1, t = setTimeout(function() {
          g = !0;
          u.abortJob(413);
        }, e || 15000);
        v.onreadystatechange = function() {
          g || (2 == v.readyState && a(27), 4 == v.readyState && (clearTimeout(t), a(29), 503 != v.status && (0 == v.status || 399 < v.status) ? u.abortJob(415, [v.status]) : 0 == v.responseText.length && c == U[0] ? u.abortJob(416) : f.call(this, v.responseText)));
        };
        v.onerror = function() {
          u.abortJob(408);
        };
      }
      v.open(c, b, !0);
      null == d ? v.send() : v.send(d);
    }
    function B(a) {
      var b = "", c = "", d;
      for (d in a[E]) {
        var e = a[E][d];
        "-" != e && (b += c + d + "=" + e + ";", c = " ");
      }
      return b;
    }
    function F(a) {
      delete D["" + a];
      localStorage.cache = pako.deflate(JSON.stringify(D), {to:"string"});
    }
    function G(a, c) {
      if (null != n) {
        try {
          if ("undefined" != typeof n.sent) {
            return;
          }
          var d = g(n, {});
          c && (d.payload = c);
          d.status = a;
          m.sendMessage(d);
          h();
        } catch (x) {
          p.reportBug(x, "abort");
        }
      }
      b.console.clear();
    }
    var J = null, n = null, K = /automated access|api-services-support@/, M = [function(a) {
    }, function(a) {
      if (null != n) {
        var b = !0;
        if (a.initiator) {
          if (a.initiator.startsWith("http")) {
            return;
          }
        } else {
          if (a.originUrl && !a.originUrl.startsWith("moz-extension")) {
            return;
          }
        }
        if (n.url == a.url) {
          P = a.frameId, W = a.tabId, Y = a.parentFrameId, b = !1;
        } else {
          if (P == a.parentFrameId || Y == a.parentFrameId || P == a.frameId) {
            b = !1;
          }
        }
        if (-2 != P && !(0 < a.tabId && W != a.tabId)) {
          a = a.requestHeaders;
          var c = {};
          if (!a.find(function(a) {
            return "krequestid" === a.name;
          })) {
            "" === n.headers.Cookie && (b = !0);
            (n.timeout + "").endsWith("108") || (n.headers.Cookie = b ? "" : V);
            for (var d in n.headers) {
              b = !1;
              for (var e = 0; e < a.length; ++e) {
                if (a[e].name.toLowerCase() == d.toLowerCase()) {
                  "" == n.headers[d] ? (a.splice(e, 1), e--) : a[e].value = n.headers[d];
                  b = !0;
                  break;
                }
              }
              b || "" == n.headers[d] || a.push({name:A ? d.toLowerCase() : d, value:n.headers[d]});
            }
            c.requestHeaders = a;
            return c;
          }
        }
      }
    }, function(a) {
      var b = a.responseHeaders;
      try {
        if (a.initiator) {
          if (a.initiator.startsWith("http")) {
            return;
          }
        } else {
          if (a.originUrl && !a.originUrl.startsWith("moz-extension")) {
            return;
          }
        }
        if (0 < a.tabId && W != a.tabId || null == n || b.find(function(a) {
          return "krequestid" === a.name;
        })) {
          return;
        }
        for (var d = (n.timeout + "").endsWith("108"), e = !1, f = [], g = 0; g < b.length; g++) {
          var k = b[g], l = k.name.toLowerCase();
          "set-cookie" == l ? (-1 < k.value.indexOf("xpires") && c.parseCookieHeader(f, k.value), d || b.splice(g--, 1)) : "x-frame-options" == l && (b.splice(g, 1), g--);
        }
        for (g = 0; g < f.length; g++) {
          var h = f[g];
          if ("undefined" == typeof D[E][h[0]] || D[E][h[0]] != h[1]) {
            e = !0, D[E][h[0]] = h[1];
          }
        }
        !d && e && n.url == a.url && (localStorage.cache = pako.deflate(JSON.stringify(D), {to:"string"}), V = B(D));
      } catch (ba) {
      }
      return {responseHeaders:b};
    }, function(a) {
      if (null != n && n.url == a.url) {
        var b = 0;
        switch(a.error) {
          case "net::ERR_TUNNEL_CONNECTION_FAILED":
            b = 510;
            break;
          case "net::ERR_INSECURE_RESPONSE":
            b = 511;
            break;
          case "net::ERR_CONNECTION_REFUSED":
            b = 512;
            break;
          case "net::ERR_BAD_SSL_CLIENT_AUTH_CERT":
            b = 513;
            break;
          case "net::ERR_CONNECTION_CLOSED":
            b = 514;
            break;
          case "net::ERR_NAME_NOT_RESOLVED":
            b = 515;
            break;
          case "net::ERR_NAME_RESOLUTION_FAILED":
            b = 516;
            break;
          case "net::ERR_ABORTED":
          case "net::ERR_CONNECTION_ABORTED":
            b = 517;
            break;
          case "net::ERR_CONTENT_DECODING_FAILED":
            b = 518;
            break;
          case "net::ERR_NETWORK_ACCESS_DENIED":
            b = 519;
            break;
          case "net::ERR_NETWORK_CHANGED":
            b = 520;
            break;
          case "net::ERR_INCOMPLETE_CHUNKED_ENCODING":
            b = 521;
            break;
          case "net::ERR_CONNECTION_TIMED_OUT":
          case "net::ERR_TIMED_OUT":
            b = 522;
            break;
          case "net::ERR_CONNECTION_RESET":
            b = 523;
            break;
          case "net::ERR_NETWORK_IO_SUSPENDED":
            b = 524;
            break;
          case "net::ERR_EMPTY_RESPONSE":
            b = 525;
            break;
          case "net::ERR_SSL_PROTOCOL_ERROR":
            b = 526;
            break;
          case "net::ERR_ADDRESS_UNREACHABLE":
            b = 527;
            break;
          case "net::ERR_INTERNET_DISCONNECTED":
            b = 528;
            break;
          case "net::ERR_BLOCKED_BY_ADMINISTRATOR":
            b = 529;
            break;
          case "net::ERR_SSL_VERSION_OR_CIPHER_MISMATCH":
            b = 530;
            break;
          case "net::ERR_CONTENT_LENGTH_MISMATCH":
            b = 531;
            break;
          case "net::ERR_PROXY_CONNECTION_FAILED":
            b = 532;
            break;
          default:
            b = 533;
            return;
        }
        setTimeout(function() {
          u.setStatTime(33);
          u.abortJob(b);
        }, 0);
      }
    }], Q = !1, X = !1, N = null, L = null, R = function() {
      X = !0;
      for (var a = 0; a < N.length; a++) {
        var c = N[a], d = window, e = 0;
        try {
          for (; e < c.path.length - 1; e++) {
            d = d[c.path[e]];
          }
          delete c.a.types;
          if (c.b) {
            d[c.path[e]](M[c.index], c.a, c.b);
          } else {
            d[c.path[e]](M[c.index], c.a);
          }
        } catch (I) {
          console.log(I);
        }
      }
      b.console.clear();
    }, U = ["GET", "HEAD", "POST", "PUT", "DELETE"], D = {}, V = "", E = 1;
    try {
      localStorage.cache && (D = JSON.parse(pako.inflate(localStorage.cache, {to:"string"})));
    } catch (t) {
      setTimeout(function() {
        p.reportBug(t, pako.inflate(localStorage.cache, {to:"string"}));
      }, 2000);
    }
    var P = -2, W = -1, Y = -2;
    return {onMessage:function(a) {
      "hhhh" == a.key && chrome.webRequest.onBeforeSendHeaders.addListener(function(a) {
        if (null != n) {
          var b = !0;
          if (a.initiator) {
            if (a.initiator.startsWith("http")) {
              return;
            }
          } else {
            if (a.originUrl && !a.originUrl.startsWith("moz-extension")) {
              return;
            }
          }
          n.url == a.url && (P = a.frameId, W = a.tabId, Y = a.parentFrameId, b = !1);
          if (-2 != P && P == a.frameId && W == a.tabId && Y == a.parentFrameId) {
            a = a.requestHeaders;
            var c = {};
            (n.timeout + "").endsWith("108") || (n.headers.Cookie = b ? "" : V);
            for (var d in n.headers) {
              b = !1;
              for (var e = 0; e < a.length; ++e) {
                if (a[e].name.toLowerCase() == d.toLowerCase()) {
                  "" == n.headers[d] ? a.splice(e, 1) : a[e].value = n.headers[d];
                  b = !0;
                  break;
                }
              }
              b || "" == n.headers[d] || a.push({name:A ? d.toLowerCase() : d, value:n.headers[d]});
            }
            c.requestHeaders = a;
            return c;
          }
        }
      }, {urls:["<all_urls>"]}, ["blocking", "requestHeaders"]);
      switch(a.key) {
        case "o0":
        case "o1":
          n = a, n.stats = {start:Date.now(), times:[]};
      }
      switch(a.key) {
        case "update":
          chrome.runtime.requestUpdateCheck(function(a, b) {
            console.log(a, b);
            "update_available" == a && chrome.runtime.reload();
          });
          break;
        case "o0":
          u.clearIframe();
          f(a);
          break;
        case "o1":
          u.clearIframe();
          d(a);
          break;
        case "o2":
          F(a.domainId);
          break;
        case "1":
          document.location.reload(!1);
      }
    }, clearIframe:h, endSession:F, getOutgoingMessage:g, setStatTime:a, getFilters:function() {
      return J;
    }, getMessage:function() {
      return n;
    }, clearMessage:function() {
      n = null;
      if (null != N && X) {
        X = !1;
        for (var a = 0; a < N.length; a++) {
          var c = N[a];
          if (c) {
            try {
              for (var d = window, e = 0; e < c.path.length - 1; e++) {
                d = d[c.path[e]];
              }
              d.removeListener(M[c.index]);
            } catch (I) {
            }
          }
        }
        b.console.clear();
      }
    }, abortJob:G};
  }();
})();

