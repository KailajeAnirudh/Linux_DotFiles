var $jscomp = $jscomp || {};
$jscomp.scope = {};
$jscomp.arrayIteratorImpl = function(c) {
  var m = 0;
  return function() {
    return m < c.length ? {done:!1, value:c[m++]} : {done:!0};
  };
};
$jscomp.arrayIterator = function(c) {
  return {next:$jscomp.arrayIteratorImpl(c)};
};
$jscomp.makeIterator = function(c) {
  var m = "undefined" != typeof Symbol && Symbol.iterator && c[Symbol.iterator];
  return m ? m.call(c) : $jscomp.arrayIterator(c);
};
$jscomp.arrayFromIterator = function(c) {
  for (var m, l = []; !(m = c.next()).done;) {
    l.push(m.value);
  }
  return l;
};
$jscomp.arrayFromIterable = function(c) {
  return c instanceof Array ? c : $jscomp.arrayFromIterator($jscomp.makeIterator(c));
};
$jscomp.checkStringArgs = function(c, m, l) {
  if (null == c) {
    throw new TypeError("The 'this' value for String.prototype." + l + " must not be null or undefined");
  }
  if (m instanceof RegExp) {
    throw new TypeError("First argument to String.prototype." + l + " must not be a regular expression");
  }
  return c + "";
};
$jscomp.ASSUME_ES5 = !1;
$jscomp.ASSUME_NO_NATIVE_MAP = !1;
$jscomp.ASSUME_NO_NATIVE_SET = !1;
$jscomp.SIMPLE_FROUND_POLYFILL = !1;
$jscomp.defineProperty = $jscomp.ASSUME_ES5 || "function" == typeof Object.defineProperties ? Object.defineProperty : function(c, m, l) {
  c != Array.prototype && c != Object.prototype && (c[m] = l.value);
};
$jscomp.getGlobal = function(c) {
  return "undefined" != typeof window && window === c ? c : "undefined" != typeof global && null != global ? global : c;
};
$jscomp.global = $jscomp.getGlobal(this);
$jscomp.polyfill = function(c, m, l, k) {
  if (m) {
    l = $jscomp.global;
    c = c.split(".");
    for (k = 0; k < c.length - 1; k++) {
      var t = c[k];
      t in l || (l[t] = {});
      l = l[t];
    }
    c = c[c.length - 1];
    k = l[c];
    m = m(k);
    m != k && null != m && $jscomp.defineProperty(l, c, {configurable:!0, writable:!0, value:m});
  }
};
$jscomp.polyfill("String.prototype.startsWith", function(c) {
  return c ? c : function(c, l) {
    var k = $jscomp.checkStringArgs(this, c, "startsWith");
    c += "";
    var m = k.length, n = c.length;
    l = Math.max(0, Math.min(l | 0, k.length));
    for (var E = 0; E < n && l < m;) {
      if (k[l++] != c[E++]) {
        return !1;
      }
    }
    return E >= n;
  };
}, "es6", "es3");
$jscomp.owns = function(c, m) {
  return Object.prototype.hasOwnProperty.call(c, m);
};
$jscomp.assign = "function" == typeof Object.assign ? Object.assign : function(c, m) {
  for (var l = 1; l < arguments.length; l++) {
    var k = arguments[l];
    if (k) {
      for (var t in k) {
        $jscomp.owns(k, t) && (c[t] = k[t]);
      }
    }
  }
  return c;
};
$jscomp.polyfill("Object.assign", function(c) {
  return c || $jscomp.assign;
}, "es6", "es3");
$jscomp.polyfill("Object.is", function(c) {
  return c ? c : function(c, l) {
    return c === l ? 0 !== c || 1 / c === 1 / l : c !== c && l !== l;
  };
}, "es6", "es3");
$jscomp.polyfill("Array.prototype.includes", function(c) {
  return c ? c : function(c, l) {
    var k = this;
    k instanceof String && (k = String(k));
    var m = k.length;
    l = l || 0;
    for (0 > l && (l = Math.max(l + m, 0)); l < m; l++) {
      var n = k[l];
      if (n === c || Object.is(n, c)) {
        return !0;
      }
    }
    return !1;
  };
}, "es7", "es3");
$jscomp.polyfill("String.prototype.includes", function(c) {
  return c ? c : function(c, l) {
    return -1 !== $jscomp.checkStringArgs(this, c, "includes").indexOf(c, l || 0);
  };
}, "es6", "es3");
var onlyStock = !1, scanner = function() {
  function c(c, l, n, m, F, D) {
    var k = new XMLHttpRequest, t = !1, f = setTimeout(function() {
      t = !0;
      D();
    }, m || 4000);
    k.onreadystatechange = function() {
      t || (clearTimeout(f), F(k));
    };
    k.onerror = D;
    k.open(l, c, !0);
    null == n ? k.send() : k.send(n);
  }
  function m(k, l) {
    var n = {};
    if (null == document.body) {
      n.status = 599, l(n);
    } else {
      if (document.body.textContent.match("you're not a robot")) {
        n.status = 403, l(n);
      } else {
        for (var m = document.evaluate("//comment()", document, null, XPathResult.ANY_TYPE, null), t = m.iterateNext(), D = ""; t;) {
          D += t, t = m.iterateNext();
        }
        if (D.match(/automated access|api-services-support@/)) {
          n.status = 403, l(n);
        } else {
          if (D.match(/ref=cs_503_link/)) {
            n.status = 503, l(n);
          } else {
            var C = 0;
            if (k.scrapeFilters && 0 < k.scrapeFilters.length) {
              m = {};
              t = null;
              var u = "", f = null, z = {}, A = {}, H = !1, y = function(a, b, e) {
                var g = [];
                if (!a.selector) {
                  if (!a.regExp) {
                    return u = "invalid selector, sel/regexp", !1;
                  }
                  var d = document.getElementsByTagName("html")[0].innerHTML.match(new RegExp(a.regExp, "i"));
                  if (!d || d.length < a.reGroup) {
                    d = "regexp fail: html - " + a.name + e;
                    if (!1 === a.optional) {
                      return u = d, !1;
                    }
                    f += " // " + d;
                    return !0;
                  }
                  return d[a.reGroup];
                }
                d = b.querySelectorAll(a.selector);
                0 == d.length && (d = b.querySelectorAll(a.altSelector));
                if (0 == d.length) {
                  if (!0 === a.optional) {
                    return !0;
                  }
                  u = "selector no match: " + a.name + e;
                  return !1;
                }
                if (a.parentSelector && (d = [d[0].parentNode.querySelector(a.parentSelector)], null == d[0])) {
                  if (!0 === a.optional) {
                    return !0;
                  }
                  u = "parent selector no match: " + a.name + e;
                  return !1;
                }
                if ("undefined" != typeof a.multiple && null != a.multiple && (!0 === a.multiple && 1 > d.length || !1 === a.multiple && 1 < d.length)) {
                  if (!H) {
                    return H = !0, y(a, b, e);
                  }
                  e = "selector multiple mismatch: " + a.name + e + " found: " + d.length;
                  if (!1 === a.optional) {
                    a = "";
                    for (var p in d) {
                      !d.hasOwnProperty(p) || 1000 < a.length || (a += " - " + p + ": " + d[p].outerHTML + " " + d[p].getAttribute("class") + " " + d[p].getAttribute("id"));
                    }
                    u = e + a + " el: " + b.getAttribute("class") + " " + b.getAttribute("id");
                    return !1;
                  }
                  f += " // " + e;
                  return !0;
                }
                if (a.isListSelector) {
                  return z[a.name] = d, !0;
                }
                if (!a.attribute) {
                  return u = "selector attribute undefined?: " + a.name + e, !1;
                }
                for (var h in d) {
                  if (d.hasOwnProperty(h)) {
                    b = d[h];
                    if (!b) {
                      break;
                    }
                    if (a.childNode) {
                      a.childNode = Number(a.childNode);
                      b = b.childNodes;
                      if (b.length < a.childNode) {
                        d = "childNodes fail: " + b.length + " - " + a.name + e;
                        if (!1 === a.optional) {
                          return u = d, !1;
                        }
                        f += " // " + d;
                        return !0;
                      }
                      b = b[a.childNode];
                    }
                    b = "text" == a.attribute ? b.textContent : "html" == a.attribute ? b.innerHTML : b.getAttribute(a.attribute);
                    if (!b || 0 == b.length || 0 == b.replace(/(\r\n|\n|\r)/gm, "").replace(/^\s+|\s+$/g, "").length) {
                      d = "selector attribute null: " + a.name + e;
                      if (!1 === a.optional) {
                        return u = d, !1;
                      }
                      f += " // " + d;
                      return !0;
                    }
                    if (a.regExp) {
                      p = b.match(new RegExp(a.regExp, "i"));
                      if (!p || p.length < a.reGroup) {
                        d = "regexp fail: " + b + " - " + a.name + e;
                        if (!1 === a.optional) {
                          return u = d, !1;
                        }
                        f += " // " + d;
                        return !0;
                      }
                      g.push(p[a.reGroup]);
                    } else {
                      g.push(b);
                    }
                    if (!a.multiple) {
                      break;
                    }
                  }
                }
                a.multiple || (g = g[0]);
                return g;
              };
              D = document;
              var a = !1, b = {}, p;
              for (p in k.scrapeFilters) {
                b.$jscomp$loop$prop$pageType$73 = p;
                a: {
                  if (a) {
                    break;
                  }
                  b.$jscomp$loop$prop$pageFilter$70 = k.scrapeFilters[b.$jscomp$loop$prop$pageType$73];
                  var h = b.$jscomp$loop$prop$pageFilter$70.pageVersionTest, g = document.querySelectorAll(h.selector);
                  0 == g.length && (g = document.querySelectorAll(h.altSelector));
                  if (0 != g.length) {
                    if ("undefined" != typeof h.multiple && null != h.multiple) {
                      if (!0 === h.multiple && 2 > g.length) {
                        break a;
                      }
                      if (!1 === h.multiple && 1 < g.length) {
                        break a;
                      }
                    }
                    if (h.attribute) {
                      var e = null;
                      e = "text" == h.attribute ? "" : g[0].getAttribute(h.attribute);
                      if (null == e) {
                        break a;
                      }
                    }
                    t = b.$jscomp$loop$prop$pageType$73;
                    g = {};
                    for (var d in b.$jscomp$loop$prop$pageFilter$70) {
                      if (a) {
                        break;
                      }
                      g.$jscomp$loop$prop$sel$66 = b.$jscomp$loop$prop$pageFilter$70[d];
                      if (g.$jscomp$loop$prop$sel$66.name != h.name) {
                        if (g.$jscomp$loop$prop$sel$66.parentList) {
                          e = [];
                          if ("undefined" != typeof z[g.$jscomp$loop$prop$sel$66.parentList]) {
                            e = z[g.$jscomp$loop$prop$sel$66.parentList];
                          } else {
                            if (!0 === y(b.$jscomp$loop$prop$pageFilter$70[g.$jscomp$loop$prop$sel$66.parentList], D, b.$jscomp$loop$prop$pageType$73)) {
                              e = z[g.$jscomp$loop$prop$sel$66.parentList];
                            } else {
                              break;
                            }
                          }
                          A[g.$jscomp$loop$prop$sel$66.parentList] || (A[g.$jscomp$loop$prop$sel$66.parentList] = []);
                          var B = 0, x = {}, v;
                          for (v in e) {
                            if (a) {
                              break;
                            }
                            if (e.hasOwnProperty(v)) {
                              if ("lager" == g.$jscomp$loop$prop$sel$66.name) {
                                B++;
                                try {
                                  var q = void 0, r = void 0;
                                  g.$jscomp$loop$prop$sel$66.selector && (q = e[v].querySelector(g.$jscomp$loop$prop$sel$66.selector));
                                  g.$jscomp$loop$prop$sel$66.altSelector && (r = e[v].querySelector(g.$jscomp$loop$prop$sel$66.altSelector));
                                  r && (r = r.getAttribute(g.$jscomp$loop$prop$sel$66.attribute));
                                  var I = 999, L = !1;
                                  try {
                                    L = -1 != e[v].textContent.toLowerCase().indexOf("add to cart to see product details.");
                                  } catch (J) {
                                  }
                                  if (!r) {
                                    try {
                                      var w = JSON.parse(g.$jscomp$loop$prop$sel$66.regExp);
                                      if (w.sel1) {
                                        try {
                                          var R = JSON.parse(e[v].querySelectorAll(w.sel1)[0].dataset[w.dataSet1]);
                                          r = R[w.val1];
                                          I = R.maxQty;
                                        } catch (J) {
                                        }
                                      }
                                      if (!r && w.sel2) {
                                        try {
                                          var M = JSON.parse(e[v].querySelectorAll(w.sel2)[0].dataset[w.dataSet2]);
                                          r = M[w.val2];
                                          I = M.maxQty;
                                        } catch (J) {
                                        }
                                      }
                                    } catch (J) {
                                    }
                                  }
                                  if (q) {
                                    C++;
                                    x.$jscomp$loop$prop$mapIndex$67 = v + "";
                                    x.$jscomp$loop$prop$busy$68 = !0;
                                    var G = document.location.href.match(/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/)[1];
                                    G = G[1];
                                    null == G || 9 > G.length || (chrome.runtime.sendMessage({type:"getStock", asin:G, sellerId:sellerId, oid:r, maxQty:I, isMAP:L, host:document.location.hostname, referer:document.location + "", domainId:k.domainId, force:!0, session:"unknown"}, function(a, b) {
                                      return function(d) {
                                        a.$jscomp$loop$prop$busy$68 && (a.$jscomp$loop$prop$busy$68 = !1, "undefined" != typeof d && (d.error ? console.log(d.error) : (A[b.$jscomp$loop$prop$sel$66.parentList][a.$jscomp$loop$prop$mapIndex$67][b.$jscomp$loop$prop$sel$66.name] = d, 0 == --C && l(n))));
                                      };
                                    }(x, g)), setTimeout(function(a) {
                                      return function() {
                                        a.$jscomp$loop$prop$busy$68 && 0 == --C && (a.$jscomp$loop$prop$busy$68 = !1, l(n));
                                      };
                                    }(x), 3000));
                                  }
                                } catch (J) {
                                }
                              } else {
                                if ("revealMAP" == g.$jscomp$loop$prop$sel$66.name) {
                                  x.$jscomp$loop$prop$revealMAP$71 = g.$jscomp$loop$prop$sel$66, q = void 0, q = x.$jscomp$loop$prop$revealMAP$71.selector ? e[v].querySelector(x.$jscomp$loop$prop$revealMAP$71.selector) : e[v], null != q && q.textContent.match(new RegExp(x.$jscomp$loop$prop$revealMAP$71.regExp, "i")) && (q = document.location.href.match(/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/), q = q[1], r = b.$jscomp$loop$prop$pageFilter$70.sellerId, "undefined" == typeof r || null == r || null == q || 
                                  2 > q.length || (r = e[v].querySelector('input[name="oid"]').value, null == r || 20 > r + 0 || (q = x.$jscomp$loop$prop$revealMAP$71.altSelector.replace("OFFERID", r).replace("ASINID", q), C++, x.$jscomp$loop$prop$mapIndex$14$72 = v + "", c(q, "GET", null, 3000, function(a, b) {
                                    return function(d) {
                                      if (4 == d.readyState) {
                                        C--;
                                        if (200 == d.status) {
                                          try {
                                            var e = d.responseText, g = a.$jscomp$loop$prop$pageFilter$70.price;
                                            if (g && g.regExp) {
                                              if (e.match(/no valid offer--/)) {
                                                A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72] || (A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72] = {}), A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72][b.$jscomp$loop$prop$revealMAP$71.name] = -1;
                                              } else {
                                                var p = e.match(new RegExp("price info--\x3e(?:.|\\n)*?" + g.regExp + "(?:.|\\n)*?\x3c!--")), h = e.match(/price info--\x3e(?:.|\n)*?(?:<span.*?size-small.*?">)([^]*?<\/span)(?:.|\n)*?\x3c!--/);
                                                if (!p || p.length < g.reGroup) {
                                                  f += " //  priceMAP regexp fail: " + (e + " - " + g.name + a.$jscomp$loop$prop$pageType$73);
                                                } else {
                                                  var c = p[g.reGroup];
                                                  A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72] || (A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72] = {});
                                                  A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72][b.$jscomp$loop$prop$revealMAP$71.name] = c;
                                                  null != h && 2 == h.length && (A[b.$jscomp$loop$prop$revealMAP$71.parentList][b.$jscomp$loop$prop$mapIndex$14$72][b.$jscomp$loop$prop$revealMAP$71.name + "Shipping"] = h[1].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                                                }
                                              }
                                            }
                                          } catch (U) {
                                          }
                                        }
                                        0 == C && l(n);
                                      }
                                    };
                                  }(b, x), function() {
                                    0 == --C && l(n);
                                  }))));
                                } else {
                                  q = y(g.$jscomp$loop$prop$sel$66, e[v], b.$jscomp$loop$prop$pageType$73);
                                  if (!1 === q) {
                                    a = !0;
                                    break;
                                  }
                                  if (!0 !== q) {
                                    if (A[g.$jscomp$loop$prop$sel$66.parentList][v] || (A[g.$jscomp$loop$prop$sel$66.parentList][v] = {}), g.$jscomp$loop$prop$sel$66.multiple) {
                                      for (var N in q) {
                                        q.hasOwnProperty(N) && !g.$jscomp$loop$prop$sel$66.keepBR && (q[N] = q[N].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                                      }
                                      q = q.join("\u271c\u271c");
                                      A[g.$jscomp$loop$prop$sel$66.parentList][v][g.$jscomp$loop$prop$sel$66.name] = q;
                                    } else {
                                      A[g.$jscomp$loop$prop$sel$66.parentList][v][g.$jscomp$loop$prop$sel$66.name] = g.$jscomp$loop$prop$sel$66.keepBR ? q : q.replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " ");
                                    }
                                  }
                                }
                              }
                            }
                            x = {$jscomp$loop$prop$busy$68:x.$jscomp$loop$prop$busy$68, $jscomp$loop$prop$mapIndex$67:x.$jscomp$loop$prop$mapIndex$67, $jscomp$loop$prop$revealMAP$71:x.$jscomp$loop$prop$revealMAP$71, $jscomp$loop$prop$mapIndex$14$72:x.$jscomp$loop$prop$mapIndex$14$72};
                          }
                        } else {
                          e = y(g.$jscomp$loop$prop$sel$66, D, b.$jscomp$loop$prop$pageType$73);
                          if (!1 === e) {
                            a = !0;
                            break;
                          }
                          if (!0 !== e) {
                            if (g.$jscomp$loop$prop$sel$66.multiple) {
                              for (var O in e) {
                                e.hasOwnProperty(O) && !g.$jscomp$loop$prop$sel$66.keepBR && (e[O] = e[O].replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                              }
                              e = e.join();
                            } else {
                              g.$jscomp$loop$prop$sel$66.keepBR || (e = e.replace(/(\r\n|\n|\r)/gm, " ").replace(/^\s+|\s+$/g, "").replace(/\s{2,}/g, " "));
                            }
                            m[g.$jscomp$loop$prop$sel$66.name] = e;
                          }
                        }
                      }
                      g = {$jscomp$loop$prop$sel$66:g.$jscomp$loop$prop$sel$66};
                    }
                    a = !0;
                  }
                }
                b = {$jscomp$loop$prop$pageFilter$70:b.$jscomp$loop$prop$pageFilter$70, $jscomp$loop$prop$pageType$73:b.$jscomp$loop$prop$pageType$73};
              }
              if (null == t) {
                u += " // no pageVersion matched", n.status = 308, n.payload = [f, u, k.dbg1 ? document.getElementsByTagName("html")[0].innerHTML : ""];
              } else {
                if ("" === u) {
                  n.payload = [f];
                  n.scrapedData = m;
                  for (var S in A) {
                    n[S] = A[S];
                  }
                } else {
                  n.status = 305, n.payload = [f, u, k.dbg2 ? document.getElementsByTagName("html")[0].innerHTML : ""];
                }
              }
            } else {
              n.status = 306;
            }
            0 == C && l(n);
          }
        }
      }
    }
  }
  var l = !0;
  window.self === window.top && (l = !1);
  window.sandboxHasRun && (l = !1);
  l && (window.sandboxHasRun = !0, window.addEventListener("message", function(c) {
    if (c.source == window.parent && c.data && (c.origin == "chrome-extension://" + chrome.runtime.id || c.origin.startsWith("moz-extension://") || c.origin.startsWith("safari-extension://"))) {
      var l = c.data.value;
      "data" == c.data.key && l.url && l.url == document.location && setTimeout(function() {
        null == document.body ? setTimeout(function() {
          m(l, function(c) {
            window.parent.postMessage({sandbox:c}, "*");
          });
        }, 1500) : m(l, function(c) {
          window.parent.postMessage({sandbox:c}, "*");
        });
      }, 800);
    }
  }, !1), window.parent.postMessage({sandbox:document.location + "", isUrlMsg:!0}, "*"));
  window.addEventListener("error", function(c, l, m, E, F) {
    "ipbakfmnjdenbmoenhicfmoojdojjjem" != chrome.runtime.id && "blfpbjkajgamcehdbehfdioapoiibdmc" != chrome.runtime.id || console.log(F);
    return !1;
  });
  return {scan:m};
}();
(function() {
  var c = !1, m = !1, l = window.opera || -1 < navigator.userAgent.indexOf(" OPR/"), k = -1 < navigator.userAgent.toLowerCase().indexOf("firefox"), t = -1 < navigator.userAgent.toLowerCase().indexOf("edge/"), n = /Apple Computer/.test(navigator.vendor) && /Safari/.test(navigator.userAgent), E = !l && !k && !t & !n, F = k ? "Firefox" : n ? "Safari" : E ? "Chrome" : l ? "Opera" : t ? "Edge" : "Unknown", D = chrome.runtime.getManifest().version, C = !1;
  try {
    C = /Android|webOS|iPhone|iPad|iPod|BlackBerry/i.test(navigator.userAgent);
  } catch (a) {
  }
  if (!window.keepaHasRun) {
    window.keepaHasRun = !0;
    var u = 0;
    chrome.runtime.onMessage.addListener(function(a, b, p) {
      switch(a.key) {
        case "updateToken":
          f.iframeStorage ? f.iframeStorage.contentWindow.postMessage({origin:"keepaContentScript", key:"updateTokenWebsite", value:a.value}, f.iframeStorage.src) : window.postMessage({origin:"keepaContentScript", key:"updateTokenWebsite", value:a.value}, "*");
      }
    });
    window.addEventListener("message", function(a) {
      if ("undefined" == typeof a.data.sandbox) {
        if ("https://keepa.com" == a.origin || "https://test.keepa.com" == a.origin || "https://dyn.keepa.com" == a.origin) {
          if (a.data.hasOwnProperty("origin") && "keepaIframe" == a.data.origin) {
            f.handleIFrameMessage(a.data.key, a.data.value, function(b) {
              try {
                a.source.postMessage({origin:"keepaContentScript", key:a.data.key, value:b, id:a.data.id}, a.origin);
              } catch (d) {
              }
            });
          } else {
            if ("string" === typeof a.data) {
              var b = a.data.split(",");
              if (2 > b.length) {
                return;
              }
              if (2 < b.length) {
                for (var p = 2, h = b.length; p < h; p++) {
                  b[1] += "," + b[p];
                }
              }
              f.handleIFrameMessage(b[0], b[1], function(b) {
                a.source.postMessage({origin:"keepaContentScript", value:b}, a.origin);
              });
            }
          }
        }
        if (a.origin.match(/^https?:\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|jp|ca|fr|es|nl|it|in|com\.mx|com\.br)/)) {
          try {
            var g = JSON.parse(a.data);
          } catch (e) {
            return;
          }
          (g = g.asin) && "null" != g && /([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/.test(g) && (g != f.ASIN ? (f.ASIN = g, f.swapIFrame()) : 0 != u ? (window.clearTimeout(u), u = 1) : u = window.setTimeout(function() {
            f.swapIFrame();
          }, 1000));
        }
      }
    });
    var f = {domain:0, iframeStorage:null, ASIN:null, tld:"", placeholder:"", cssFlex:function() {
      var a = "flex", b = ["flex", "-webkit-flex", "-moz-box", "-webkit-box", "-ms-flexbox"], f = document.createElement("flexelement"), h;
      for (h in b) {
        try {
          if ("undefined" != f.style[b[h]]) {
            a = b[h];
            break;
          }
        } catch (g) {
        }
      }
      return a;
    }(), getDomain:function(a) {
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
        case "jp":
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
          return -1;
      }
    }, revealWorking:!1, juvecOnlyOnce:!1, revealMapOnlyOnce:!1, revealCache:{}, revealMAP:function() {
      f.revealMapOnlyOnce || (f.revealMapOnlyOnce = !0, chrome.runtime.sendMessage({type:"isPro"}, function(a) {
        if (null === a.value) {
          console.log("stock data fail");
        } else {
          var b = a.stockData, p = !0 === a.value, h = function(a) {
            a = a.trim();
            var e = b.amazonNames[a];
            return e ? "W" === e ? b.warehouseIds[f.domain] : "A" === e ? b.amazonIds[f.domain] : e : (a = a.match(new RegExp(b.sellerId))) && a[1] ? a[1] : null;
          };
          chrome.storage.local.get("revealStock", function(a) {
            "undefined" == typeof a && (a = {});
            var e = !0;
            try {
              e = "0" != a.revealStock;
            } catch (Q) {
            }
            console.log("keepa stock active: " + p + " " + e);
            try {
              if ((e || "com" == f.tld) && !f.revealWorking) {
                if (f.revealWorking = !0, document.getElementById("keepaMAP")) {
                  f.revealWorking = !1;
                } else {
                  var d = function() {
                    var a = new MutationObserver(function(b) {
                      setTimeout(function() {
                        f.revealMAP();
                      }, 100);
                      try {
                        a.disconnect();
                      } catch (T) {
                      }
                    });
                    a.observe(document.getElementById("keepaMAP").parentNode.parentNode.parentNode, {childList:!0, subtree:!0});
                  }, g = function(a, b, d, e, g, h, B, k) {
                    if ("undefined" == typeof f.revealCache[e] || null == a.parentElement.querySelector(".keepaStock")) {
                      null == k && (k = l[f.domain]);
                      var m = "" == a.id && "aod-pinned-offer" == a.parentNode.id;
                      h = h || m;
                      try {
                        d = d || -1 != a.textContent.toLowerCase().indexOf("add to cart to see product details.") || !h && /(our price|always remove it|add this item to your cart|see product details in cart|see price in cart)/i.test(document.getElementById("price").textContent);
                      } catch (V) {
                      }
                      if (d || p) {
                        c(a, b, d, e, h);
                        var v = function(a) {
                          var b = document.getElementById("keepaStock" + e);
                          if (null != b) {
                            b.innerHTML = "";
                            if (null != a && null != a.price && d) {
                              var g = document.createElement("div");
                              a = 5 == f.domain ? a.price : (Number(a.price) / 100).toFixed(2);
                              var c = new Intl.NumberFormat(" en-US en-GB de-DE fr-FR ja-JP en-CA zh-CN it-IT es-ES hi-IN es-MX pt-BR en-AU nl-NL tr-TR".split(" ")[f.domain], {style:"currency", currency:" USD GBP EUR EUR JPY CAD CNY EUR EUR INR MXN BRL AUD EUR TRY".split(" ")[f.domain]});
                              0 < a && (g.innerHTML = 'Price&emsp;&ensp;<span style="font-weight: bold;">' + c.format(a) + "</span>");
                              b.parentNode.parentNode.parentNode.prepend(g);
                            }
                            p && (a = f.revealCache[e].stock, 999 == a ? a = "999+" : 1000 == a ? a = "1000+" : f.revealCache[e].isMaxQty && 30 == a && (a += "+"), g = document.createElement("span"), g.style = "font-weight: bold;", g.innerText = a + " ", a = document.createElement("span"), a.style = "color: #dedede;", a.innerText = " (revealed by \u271c Keepa)", c = document.createElement("span"), c.style = "color:#da4c33;", c.innerText = " order limit", b.appendChild(g), f.revealCache[e].limit && 
                            (0 < f.revealCache[e].orderLimit && (c.innerText += ": " + f.revealCache[e].orderLimit), b.appendChild(c)), h && b.appendChild(a));
                          }
                        };
                        "undefined" != typeof f.revealCache[e] && -1 != f.revealCache[e] ? "pending" != f.revealCache[e] && v(f.revealCache[e]) : (f.revealCache[e] = "pending", chrome.runtime.sendMessage({type:"getStock", asin:b, oid:e, sellerId:k, maxQty:B, isMAP:d, host:document.location.hostname, force:d, referer:document.location + "", domainId:f.domain, cachedStock:f.revealCache[k], session:g}, function(a) {
                          if ("undefined" != typeof a && null != a) {
                            if (a.error) {
                              var b = document.getElementById("keepaStock" + e);
                              b.innerHTML = "";
                              var d = document.createElement("span");
                              d.style = "color:#e8c7c1;";
                              d.innerText = "error(" + a.errorCode + ")";
                              d.title = a.error + ". Contact info@keepa.com with a screenshot & URL for assistance.";
                              b.appendChild(d);
                              console.log(a.error);
                            } else {
                              f.revealCache[e] = a, f.revealCache[k] = a, v(a);
                            }
                          }
                        }));
                      }
                    }
                  }, c = function(a, b, e, g, h) {
                    b = "" == a.id && "aod-pinned-offer" == a.parentNode.id;
                    var c = (h ? a.parentElement : a).querySelector(".keepaMAP");
                    if (null == (h ? a.parentElement : a).querySelector(".keepaStock")) {
                      null != c && null != c.parentElement && c.parentElement.remove();
                      var B = h ? "165px" : "55px;height:20px;";
                      c = document.createElement("div");
                      c.id = "keepaMAP" + (h ? e + g : "");
                      c.className = "a-section a-spacing-none a-spacing-top-micro aod-clear-float keepaStock";
                      e = document.createElement("div");
                      e.className = "a-fixed-left-grid";
                      var l = document.createElement("div");
                      l.style = "padding-left:" + B;
                      h && (l.className = "a-fixed-left-grid-inner");
                      var k = document.createElement("div");
                      k.style = "width:" + B + ";margin-left:-" + B + ";float:left;";
                      k.className = "a-fixed-left-grid-col aod-padding-right-10 a-col-left";
                      B = document.createElement("div");
                      B.style = "padding-left:0%;float:left;";
                      B.className = "a-fixed-left-grid-col a-col-right";
                      var v = document.createElement("span");
                      v.className = "a-size-small a-color-tertiary";
                      var m = document.createElement("span");
                      m.style = "color: #dedede;";
                      m.innerText = "loading\u2026";
                      var x = document.createElement("span");
                      x.className = "a-size-small a-color-base";
                      x.id = "keepaStock" + g;
                      x.appendChild(m);
                      B.appendChild(x);
                      k.appendChild(v);
                      l.appendChild(k);
                      l.appendChild(B);
                      e.appendChild(l);
                      c.appendChild(e);
                      v.className = "a-size-small a-color-tertiary";
                      f.revealWorking = !1;
                      p && (v.innerText = "Stock");
                      h ? b ? (a = document.querySelector("#aod-pinned-offer-show-more-link"), 0 == a.length && document.querySelector("#aod-pinned-offer-main-content-show-more"), a.prepend(c)) : a.parentNode.insertBefore(c, a.parentNode.children[a.parentNode.children.length - 1]) : a.appendChild(c);
                      h || d();
                    }
                  }, l = "1 ATVPDKIKX0DER A3P5ROKL5A1OLE A3JWKAKR8XB7XF A1X6FK5RDHNB96 AN1VRQENFRJN5 A3DWYIK6Y9EEQB A1AJ19PSB66TGU A11IL2PNWYJU7H A1AT7YVPFBWXBL A3P5ROKL5A1OLE AVDBXBAVVSXLQ A1ZZFT5FULY4LN ANEGB3WVEVKZB A17D2BRD4YMT0X".split(" "), k = document.location.href, r = new MutationObserver(function(a) {
                    try {
                      var d = document.querySelectorAll("#aod-offer,#aod-pinned-offer");
                      if (null != d && 0 != d.length) {
                        a = null;
                        var e = d[0].querySelector('input[name="session-id"]');
                        if (e) {
                          a = e.getAttribute("value");
                        } else {
                          if (e = document.querySelector("#session-id")) {
                            a = document.querySelector("#session-id").value;
                          }
                        }
                        if (!a) {
                          for (var p = document.querySelectorAll("script"), c = $jscomp.makeIterator(p), B = c.next(); !B.done; B = c.next()) {
                            var l = B.value.text.match("ue_sid.?=.?'([0-9-]{19})'");
                            l && (a = l[1]);
                          }
                        }
                        if (a) {
                          for (var v in d) {
                            if (d.hasOwnProperty(v)) {
                              var m = d[v];
                              if (null != m && "DIV" == m.nodeName) {
                                e = void 0;
                                p = 999;
                                var x = m.querySelector('input[name="offeringID.1"]');
                                if (x) {
                                  e = x.getAttribute("value");
                                } else {
                                  try {
                                    var q = JSON.parse(m.querySelectorAll("[data-aod-atc-action]")[0].dataset.aodAtcAction);
                                    e = q.oid;
                                    p = q.maxQty;
                                  } catch (P) {
                                    try {
                                      var r = JSON.parse(m.querySelectorAll("[data-aw-aod-cart-api]")[0].dataset.awAodCartApi);
                                      e = r.oid;
                                      p = r.maxQty;
                                    } catch (W) {
                                    }
                                  }
                                }
                                if (e) {
                                  var n = m.children[0];
                                  c = null;
                                  if (b) {
                                    for (B = 0; B < b.soldByOffers.length; B++) {
                                      var w = m.querySelector(b.soldByOffers[B]);
                                      if (null != w) {
                                        var t = w.getAttribute("href");
                                        null == t && (t = w.innerHTML);
                                        c = h(t);
                                        if (null != c) {
                                          break;
                                        }
                                      }
                                    }
                                  }
                                  var u = -1 != m.textContent.toLowerCase().indexOf("add to cart to see product details.");
                                  g(n, f.ASIN, u, e, a, !0, p, c);
                                }
                              }
                            }
                          }
                        } else {
                          console.error("missing sessionId");
                        }
                      }
                    } catch (P) {
                      console.log(P), f.reportBug(P, "MAP error: " + k);
                    }
                  });
                  r.observe(document.querySelector("body"), {childList:!0, attributes:!1, characterData:!1, subtree:!0, attributeOldValue:!1, characterDataOldValue:!1});
                  window.onunload = function K() {
                    try {
                      window.detachEvent("onunload", K), r.disconnect();
                    } catch (T) {
                    }
                  };
                  var m = document.querySelector(b.soldOfferId);
                  a = null;
                  if (b) {
                    var n = document.querySelector(b.soldByBBForm);
                    n && (a = n.getAttribute("value"));
                    if (null == a) {
                      for (n = 0; n < b.soldByBB.length; n++) {
                        var w = document.querySelector(b.soldByBB[n]);
                        if (null != w && (a = h(w.innerHTML), null != a)) {
                          break;
                        }
                      }
                    }
                  }
                  if (null != m && null != m.value) {
                    var t = m.parentElement.querySelector("#session-id"), y = m.parentElement.querySelector("#ASIN"), z = m.parentElement.querySelector("#selectQuantity #quantity > option:last-child");
                    if (null != t && null != y) {
                      for (w = 0; w < b.mainEl.length; w++) {
                        var A = document.querySelector(b.mainEl[w]);
                        if (null != A) {
                          w = !1;
                          if (null != z) {
                            try {
                              w = Number(z.value);
                            } catch (K) {
                              console.log(K);
                            }
                          }
                          g(A, y.value, !1, m.value, t.value, !1, w, a);
                          break;
                        }
                      }
                    }
                  }
                  var u = document.getElementById("price");
                  if (null != u && /(our price|always remove it|add this item to your cart|see product details in cart|see price in cart)/i.test(u.textContent)) {
                    var C = document.getElementById("merchant-info");
                    t = m = "";
                    if (C) {
                      if (-1 == C.textContent.toLowerCase().indexOf("amazon.c")) {
                        var D = u.querySelector('span[data-action="a-modal"]');
                        if (D) {
                          var E = D.getAttribute("data-a-modal");
                          E.match(/offeringID\.1=(.*?)&amp/) && (m = RegExp.$1);
                        }
                        if (0 == m.length) {
                          if (E.match('map_help_pop_(.*?)"')) {
                            t = RegExp.$1;
                          } else {
                            f.revealWorking = !1;
                            return;
                          }
                        }
                      }
                      if (null != m && 10 < m.length) {
                        var F = document.querySelector("#session-id");
                        g(u, f.ASIN, !1, m, F.value, !1, !1, t);
                      }
                    } else {
                      f.revealWorking = !1;
                    }
                  } else {
                    f.revealWorking = !1;
                  }
                }
              }
            } catch (Q) {
              f.revealWorking = !1, console.log(Q);
            }
          });
        }
      }));
    }, onPageLoad:function() {
      f.tld = RegExp.$2;
      var a = RegExp.$4;
      f.ASIN || (f.ASIN = a);
      f.domain = f.getDomain(f.tld);
      chrome.storage.local.get(["s_boxType", "s_boxOfferListing"], function(a) {
        "undefined" == typeof a && (a = {});
        var b = 0 < document.location.href.indexOf("/offer-listing/");
        b && "0" === a.s_boxOfferListing && (onlyStock = !0);
        document.addEventListener("DOMContentLoaded", function(p) {
          p = document.getElementsByTagName("head")[0];
          var g = document.createElement("script");
          g.type = "text/javascript";
          g.src = chrome.runtime.getURL("chrome/content/selectionHook.js");
          p.appendChild(g);
          "0" == a.s_boxType ? f.swapIFrame() : f.getPlaceholderAndInsertIFrame(function(a, d) {
            if (void 0 !== a) {
              d = document.createElement("div");
              d.setAttribute("id", "keepaButton");
              d.setAttribute("style", "    background-color: #444;\n    border: 0 solid #ccc;\n    border-radius: 6px 6px 6px 6px;\n    color: #fff;\n    cursor: pointer;\n    font-size: 12px;\n    margin: 15px;\n    padding: 6px;\n    text-decoration: none;\n    text-shadow: none;\n    display: flex;\n    box-shadow: 0px 0px 7px 0px #888;\n    width: 100px;\n    background-repeat: no-repeat;\n    height: 32px;\n    background-position-x: 7px;\n    background-position-y: 7px;\n    text-align: center;\n    background-image: url(https://cdn.keepa.com/img/logo_circled_w.svg);\n    background-size: 80px;");
              var e = document.createElement("style");
              e.appendChild(document.createTextNode("#keepaButton:hover{background-color:#666 !important}"));
              document.head.appendChild(e);
              d.addEventListener("click", function() {
                var a = document.getElementById("keepaButton");
                a.parentNode.removeChild(a);
                f.swapIFrame();
              }, !1);
              b && (a = document.getElementById("olpTabContent"), a || (a = document.getElementById("olpProduct"), a = a.nextSibling));
              a.parentNode.insertBefore(d, a);
            }
          });
        }, !1);
      });
    }, swapIFrame:function() {
      if (onlyStock || "com.au" == f.tld) {
        try {
          f.revealMAP(document, f.ASIN, f.tld), f.revealMapOnlyOnce = !1;
        } catch (b) {
        }
      } else {
        if (!document.getElementById("keepaButton")) {
          f.swapIFrame.swapTimer && clearTimeout(f.swapIFrame.swapTimer);
          f.swapIFrame.swapTimer = setTimeout(function() {
            if (!C) {
              document.getElementById("keepaContainer") || f.getPlaceholderAndInsertIFrame(f.insertIFrame);
              try {
                f.revealMAP(document, f.ASIN, f.tld), f.revealMapOnlyOnce = !1;
              } catch (b) {
              }
              f.swapIFrame.swapTimer = setTimeout(function() {
                document.getElementById("keepaContainer") || f.getPlaceholderAndInsertIFrame(f.insertIFrame);
              }, 2000);
            }
          }, 2000);
          var a = document.getElementById("keepaContainer");
          if (null != f.iframeStorage && a) {
            try {
              f.iframeStorage.contentWindow.postMessage({origin:"keepaContentScript", key:"updateASIN", value:{d:f.domain, a:f.ASIN}}, f.iframeStorage.src);
            } catch (b) {
              console.error(b);
            }
          } else {
            f.getPlaceholderAndInsertIFrame(f.insertIFrame);
            try {
              f.revealMAP(document, f.ASIN, f.tld), f.revealMapOnlyOnce = !1;
            } catch (b) {
            }
          }
        }
      }
    }, getDevicePixelRatio:function() {
      var a = 1;
      void 0 !== window.screen.systemXDPI && void 0 !== window.screen.logicalXDPI && window.screen.systemXDPI > window.screen.logicalXDPI ? a = window.screen.systemXDPI / window.screen.logicalXDPI : void 0 !== window.devicePixelRatio && (a = window.devicePixelRatio);
      return a;
    }, getPlaceholderAndInsertIFrame:function(a) {
      chrome.storage.local.get("keepaBoxPlaceholder keepaBoxPlaceholderBackup keepaBoxPlaceholderBackupClass keepaBoxPlaceholderAppend keepaBoxPlaceholderBackupAppend webGraphType webGraphRange".split(" "), function(b) {
        "undefined" == typeof b && (b = {});
        var p = 0, c = function() {
          if (!document.getElementById("keepaButton") && !document.getElementById("amazonlive-homepage-widget")) {
            if (C) {
              var g = document.querySelector("#tabular_feature_div,#olpLinkWidget_feature_div,#tellAFriendBox_feature_div");
              try {
                document.querySelector("#keepaMobileContainer")[0].remove();
              } catch (r) {
              }
              if (g && g.previousSibling) {
                try {
                  var e = b.webGraphType;
                  try {
                    e = JSON.parse(e);
                  } catch (r) {
                  }
                  var d = b.webGraphRange;
                  try {
                    d = Number(d);
                  } catch (r) {
                  }
                  var h = Math.min(1800, 1.6 * window.innerWidth).toFixed(0), l = "https://graph.keepa.com/pricehistory.png?type=2&asin=" + f.ASIN + "&domain=" + f.domain + "&width=" + h + "&height=450";
                  l = "undefined" == typeof e ? l + "&amazon=1&new=1&used=1&salesrank=1&range=365" : l + ("&amazon=" + e[0] + "&new=" + e[1] + "&used=" + e[2] + "&salesrank=" + e[3] + "&range=" + d + "&fba=" + e[10] + "&fbm=" + e[7] + "&bb=" + e[18] + "&ld=" + e[8] + "&wd=" + e[9]);
                  var m = document.createElement("div");
                  m.setAttribute("id", "keepaMobileContainer");
                  m.setAttribute("style", "margin-bottom: 20px;");
                  var k = document.createElement("img");
                  k.setAttribute("style", "margin: 5px 0; width: " + Math.min(1800, window.innerWidth) + "px;");
                  k.setAttribute("id", "keepaImageContainer" + f.ASIN);
                  k.setAttribute("src", l);
                  document.createElement("div").setAttribute("style", "margin: 20px; display: flex;justify-content: space-evenly;");
                  m.appendChild(k);
                  g.after(m);
                  k.addEventListener("click", function() {
                    k.remove();
                    f.insertIFrame(g.previousSibling, !1, !0);
                  }, !1);
                } catch (r) {
                  console.error(r);
                }
                return;
              }
            }
            if ((e = document.getElementById("gpdp-btf-container")) && e.previousElementSibling) {
              f.insertIFrame(e.previousElementSibling, !1, !0);
            } else {
              if ((e = document.getElementsByClassName("mocaGlamorContainer")[0]) || (e = document.getElementById("dv-sims")), e || (e = document.getElementById("mas-terms-of-use")), e && e.nextSibling) {
                f.insertIFrame(e.nextSibling, !1, !0);
              } else {
                if (d = b.keepaBoxPlaceholder || "#bottomRow", e = !1, d = document.querySelector(d)) {
                  "sims_fbt" == d.previousElementSibling.id && (d = d.previousElementSibling, "bucketDivider" == d.previousElementSibling.className && (d = d.previousElementSibling), e = !0), 1 == b.keepaBoxPlaceholderAppend && (d = d.nextSibling), a(d, e);
                } else {
                  if (d = b.keepaBoxPlaceholderBackup || "#elevatorBottom", "ATFCriticalFeaturesDataContainer" == d && (d = "#ATFCriticalFeaturesDataContainer"), d = document.querySelector(d)) {
                    1 == b.keepaBoxPlaceholderBackupAppend && (d = d.nextSibling), a(d, !0);
                  } else {
                    if (d = document.getElementById("hover-zoom-end")) {
                      a(d, !0);
                    } else {
                      if (d = b.keepaBoxPlaceholderBackupClass || ".a-fixed-left-grid", (d = document.querySelector(d)) && d.nextSibling) {
                        a(d.nextSibling, !0);
                      } else {
                        e = 0;
                        d = document.getElementsByClassName("twisterMediaMatrix");
                        h = !!document.getElementById("dm_mp3Player");
                        if ((d = 0 == d.length ? document.getElementById("handleBuy") : d[0]) && 0 == e && !h && null != d.nextElementSibling) {
                          l = !1;
                          for (h = d; h;) {
                            if (h = h.parentNode, "table" === h.tagName.toLowerCase()) {
                              if ("buyboxrentTable" === h.className || /buyBox/.test(h.className) || "buyingDetailsGrid" === h.className) {
                                l = !0;
                              }
                              break;
                            } else {
                              if ("html" === h.tagName.toLowerCase()) {
                                break;
                              }
                            }
                          }
                          if (!l) {
                            d = d.nextElementSibling;
                            a(d, !1);
                            return;
                          }
                        }
                        d = document.getElementsByClassName("bucketDivider");
                        0 == d.length && (d = document.getElementsByClassName("a-divider-normal"));
                        if (!d[e]) {
                          if (!d[0]) {
                            40 > p++ && window.setTimeout(function() {
                              c();
                            }, 100);
                            return;
                          }
                          e = 0;
                        }
                        for (h = d[e]; h && d[e];) {
                          if (h = h.parentNode, "table" === h.tagName.toLowerCase()) {
                            if ("buyboxrentTable" === h.className || /buyBox/.test(h.className) || "buyingDetailsGrid" === h.className) {
                              h = d[++e];
                            } else {
                              break;
                            }
                          } else {
                            if ("html" === h.tagName.toLowerCase()) {
                              break;
                            }
                          }
                        }
                        f.placeholder = d[e];
                        d[e] && d[e].parentNode && (e = document.getElementsByClassName("lpo")[0] && d[1] && 0 == e ? d[1] : d[e], a(e, !1));
                      }
                    }
                  }
                }
              }
            }
          }
        };
        c();
      });
    }, getAFComment:function(a) {
      for (a = [a]; 0 < a.length;) {
        for (var b = a.pop(), f = 0; f < b.childNodes.length; f++) {
          var h = b.childNodes[f];
          if (8 === h.nodeType && -1 < h.textContent.indexOf("MarkAF")) {
            return h;
          }
          a.push(h);
        }
      }
      return null;
    }, getIframeUrl:function(a, b) {
      return "https://keepa.com/iframe_addon.html#" + a + "-0-" + b;
    }, insertIFrame:function(a, b) {
      if (null != f.iframeStorage && document.getElementById("keepaContainer")) {
        f.swapIFrame();
      } else {
        var p = document.getElementById("hover-zoom-end"), h = function(a) {
          for (var b = document.getElementById(a), d = []; b;) {
            d.push(b), b.id = "a-different-id", b = document.getElementById(a);
          }
          for (b = 0; b < d.length; ++b) {
            d[b].id = a;
          }
          return d;
        }("hover-zoom-end");
        chrome.storage.local.get("s_boxHorizontal", function(g) {
          "undefined" == typeof g && (g = {});
          if (null == a) {
            setTimeout(function() {
              f.getPlaceholderAndInsertIFrame(f.insertIFrame);
            }, 2000);
          } else {
            var e = g.s_boxHorizontal, d = window.innerWidth - 50;
            if (!document.getElementById("keepaContainer")) {
              g = 0 < document.location.href.indexOf("/offer-listing/");
              var c = document.createElement("div");
              "0" != e || g ? c.setAttribute("style", "min-width: 935px; width: calc(100% - 30px); height: 500px; display: flex; border:0 none; margin: 10px 0 0;") : (d -= 550, 960 > d && (d = 960), c.setAttribute("style", "min-width: 935px; max-width:" + d + "px;display: flex;  height: 500px; border:0 none; margin: 10px 0 0;"));
              c.setAttribute("id", "keepaContainer");
              var l = document.createElement("iframe");
              e = document.createElement("div");
              e.setAttribute("id", "keepaClear");
              l.setAttribute("style", "width: 100%; height: 100%; border:0 none;overflow: hidden;");
              l.setAttribute("src", "https://keepa.com/keepaBox.html");
              l.setAttribute("scrolling", "no");
              l.setAttribute("id", "keepa");
              m || (m = !0);
              c.appendChild(l);
              d = !1;
              if (!b) {
                null == a.parentNode || "promotions_feature_div" !== a.parentNode.id && "dp-out-of-stock-top_feature_div" !== a.parentNode.id || (a = a.parentNode);
                try {
                  var k = a.previousSibling.previousSibling;
                  null != k && "technicalSpecifications_feature_div" == k.id && (a = k);
                } catch (M) {
                }
                0 < h.length && (p = h[h.length - 1]) && "centerCol" != p.parentElement.id && ((k = f.getFirstInDOM([a, p], document.body)) && 600 < k.parentElement.offsetWidth && (a = k), a === p && (d = !0));
                (k = document.getElementById("title") || document.getElementById("title_row")) && f.getFirstInDOM([a, k], document.body) !== k && (a = k);
              }
              k = document.getElementById("vellumMsg");
              null != k && (a = k);
              k = document.body;
              var q = document.documentElement;
              q = Math.max(k.scrollHeight, k.offsetHeight, q.clientHeight, q.scrollHeight, q.offsetHeight);
              var r = a.offsetTop / q;
              if (0.5 < r || 0 > r) {
                k = f.getAFComment(k), null != k && (r = a.offsetTop / q, 0.5 > r && (a = k));
              }
              if (a.parentNode) {
                k = document.querySelector(".container_vertical_middle");
                g ? (a = document.getElementById("olpTabContent"), a || (a = document.getElementById("olpProduct"), a = a.nextSibling), a.parentNode.insertBefore(c, a)) : "burjPageDivider" == a.id ? (a.parentNode.insertBefore(c, a), b || a.parentNode.insertBefore(e, c.nextSibling)) : "bottomRow" == a.id ? (a.parentNode.insertBefore(c, a), b || a.parentNode.insertBefore(e, c.nextSibling)) : d ? (a.parentNode.insertBefore(c, a.nextSibling), b || a.parentNode.insertBefore(e, c.nextSibling)) : null != 
                k ? (a = k, a.parentNode.insertBefore(c, a.nextSibling), b || a.parentNode.insertBefore(e, c.nextSibling)) : (a.parentNode.insertBefore(c, a), b || a.parentNode.insertBefore(e, c));
                f.iframeStorage = l;
                c.style.display = f.cssFlex;
                var n = !1, t = 5;
                if (!C) {
                  var w = setInterval(function() {
                    if (0 >= t--) {
                      clearInterval(w);
                    } else {
                      var a = null != document.getElementById("keepa" + f.ASIN);
                      try {
                        if (!a) {
                          throw f.getPlaceholderAndInsertIFrame(f.insertIFrame), 1;
                        }
                        if (n) {
                          throw 1;
                        }
                        document.getElementById("keepa" + f.ASIN).contentDocument.location = iframeUrl;
                      } catch (G) {
                        clearInterval(w);
                      }
                    }
                  }, 4000), u = function() {
                    n = !0;
                    l.removeEventListener("load", u, !1);
                    f.synchronizeIFrame();
                  };
                  l.addEventListener("load", u, !1);
                }
              } else {
                f.swapIFrame();
              }
            }
          }
        });
      }
    }, handleIFrameMessage:function(a, b, p) {
      switch(a) {
        case "resize":
          c || (c = !0);
          b = "" + b;
          -1 == b.indexOf("px") && (b += "px");
          if (a = document.getElementById("keepaContainer")) {
            a.style.height = b;
          }
          break;
        case "ping":
          p({location:chrome.runtime.id + " " + document.location});
          break;
        case "openPage":
          chrome.runtime.sendMessage({type:"openPage", url:b});
          break;
        case "getToken":
          f.sendMessageWithRetry({type:"getCookie", key:"token"}, 3, 2000, function(a) {
            p({token:a.value, install:a.install, d:f.domain, a:f.ASIN});
          }, function(a) {
            p({d:f.domain, a:f.ASIN});
          });
          break;
        case "setCookie":
          chrome.runtime.sendMessage({type:"setCookie", key:b.key, val:b.val});
      }
    }, sendMessageWithRetry:function(a, b, c, f, g) {
      var e = 0, d = !1, h = function() {
        e += 1;
        chrome.runtime.sendMessage(a, function(a) {
          d || (d = !0, f(a));
        });
        setTimeout(function() {
          d || (e < b ? (d = !0, setTimeout(h, c)) : (console.log("Failed to receive a response after maximum retries."), g()));
        }, c);
      };
      h();
    }, synchronizeIFrame:function() {
      var a = 0;
      chrome.storage.local.get("s_boxHorizontal", function(b) {
        "undefined" != typeof b && "undefined" != typeof b.s_boxHorizontal && (a = b.s_boxHorizontal);
      });
      var b = window.innerWidth, c = !1;
      C || window.addEventListener("resize", function() {
        c || (c = !0, window.setTimeout(function() {
          if (b != window.innerWidth && "0" == a) {
            b = window.innerWidth;
            var f = window.innerWidth - 50;
            f -= 550;
            935 > f && (f = 935);
            document.getElementById("keepaContainer").style.width = f;
          }
          c = !1;
        }, 100));
      }, !1);
    }, getFirstInDOM:function(a, b) {
      var c;
      for (b = b.firstChild; b; b = b.nextSibling) {
        if ("IFRAME" !== b.nodeName && 1 === b.nodeType) {
          if (-1 !== a.indexOf(b)) {
            return b;
          }
          if (c = f.getFirstInDOM(a, b)) {
            return c;
          }
        }
      }
      return null;
    }, getClipRect:function(a) {
      "string" === typeof a && (a = document.querySelector(a));
      var b = 0, c = 0, h = function(a) {
        b += a.offsetLeft;
        c += a.offsetTop;
        a.offsetParent && h(a.offsetParent);
      };
      h(a);
      return 0 == c && 0 == b ? f.getClipRect(a.parentNode) : {top:c, left:b, width:a.offsetWidth, height:a.offsetHeight};
    }, findPlaceholderBelowImages:function(a) {
      var b = a, c, h = 100;
      do {
        for (h--, c = null; !c;) {
          c = a.nextElementSibling, c || (c = a.parentNode.nextElementSibling), a = c ? c : a.parentNode.parentNode, !c || "IFRAME" !== c.nodeName && "SCRIPT" !== c.nodeName && 1 === c.nodeType || (c = null);
        }
      } while (0 < h && 100 < f.getClipRect(c).left);
      return c ? c : b;
    }, httpGet:function(a, b) {
      var c = new XMLHttpRequest;
      b && (c.onreadystatechange = function() {
        4 == c.readyState && b.call(this, c.responseText);
      });
      c.open("GET", a, !0);
      c.send();
    }, httpPost2:function(a, b, c, f, g) {
      var e = new XMLHttpRequest;
      f && (e.onreadystatechange = function() {
        4 == e.readyState && f.call(this, e.responseText);
      });
      e.withCredentials = g;
      e.open("POST", a, !0);
      e.setRequestHeader("Content-Type", c);
      e.send(b);
    }, httpPost:function(a, b, c, h) {
      f.httpPost2(a, b, "text/plain;charset=UTF-8", c, h);
    }, lastBugReport:0, reportBug:function(a, b, c) {
      var h = Date.now();
      if (!(6E5 > h - f.lastBugReport || /(dead object)|(Script error)|(\.location is null)/i.test(a))) {
        f.lastBugReport = h;
        h = "";
        try {
          h = Error().stack.split("\n").splice(1).splice(1).join("&ensp;&lArr;&ensp;");
          if (!/(keepa|content)\.js/.test(h)) {
            return;
          }
          h = h.replace(/chrome-extension:\/\/.*?\/content\//g, "").replace(/:[0-9]*?\)/g, ")").replace(/[ ]{2,}/g, "");
        } catch (g) {
        }
        if ("object" == typeof a) {
          try {
            a = a instanceof Error ? a.toString() : JSON.stringify(a);
          } catch (g) {
          }
        }
        null == c && (c = {exception:a, additional:b, url:document.location.host, stack:h});
        null != c.url && c.url.startsWith("blob:") || (c.keepaType = E ? "keepaChrome" : l ? "keepaOpera" : n ? "keepaSafari" : t ? "keepaEdge" : "keepaFirefox", c.version = D, chrome.storage.local.get("token", function(a) {
          "undefined" == typeof a && (a = {token:"undefined"});
          f.httpPost("https://dyn.keepa.com/service/bugreport/?user=" + a.token + "&type=" + F, JSON.stringify(c));
        }));
      }
    }};
    window.onerror = function(a, b, c, h, g) {
      if ("string" !== typeof a) {
        g = a.error;
        var e = a.filename || a.fileName;
        c = a.lineno || a.lineNumber;
        h = a.colno || a.columnNumber;
        a = a.message || a.name || g.message || g.name;
      }
      a = a.toString();
      var d = "";
      h = h || 0;
      if (g && g.stack) {
        d = g.stack;
        try {
          d = g.stack.split("\n").splice(1).splice(1).join("&ensp;&lArr;&ensp;");
          if (!/(keepa|content)\.js/.test(d)) {
            return;
          }
          d = d.replace(/chrome-extension:\/\/.*?\/content\//g, "").replace(/:[0-9]*?\)/g, ")").replace(/[ ]{2,}/g, "");
        } catch (B) {
        }
      }
      "undefined" === typeof c && (c = 0);
      "undefined" === typeof h && (h = 0);
      a = {msg:a, url:(b || e || document.location.toString()) + ":" + c + ":" + h, stack:d};
      "ipbakfmnjdenbmoenhicfmoojdojjjem" != chrome.runtime.id && "blfpbjkajgamcehdbehfdioapoiibdmc" != chrome.runtime.id || console.log(a);
      f.reportBug(null, null, a);
      return !1;
    };
    if (window.self == window.top && (document.addEventListener("DOMContentLoaded", function(a) {
      chrome.runtime.sendMessage({type:"optionalPermissionsRequired"}, function(a) {
        if (!0 === a.value) {
          var b = 0;
          console.log("opr: ", a.value);
          var c = function() {
            10 < b++ && document.body.removeEventListener("click", c);
            chrome.runtime.sendMessage({type:"optionalPermissions"}, function(a) {
              document.body.removeEventListener("click", c);
            });
          };
          document.body.addEventListener("click", c);
        }
      });
    }), !(/.*music\.amazon\..*/.test(document.location.href) || /.*primenow\.amazon\..*/.test(document.location.href) || /.*amazonlive-portal\.amazon\..*/.test(document.location.href) || /.*amazon\.com\/restaurants.*/.test(document.location.href)))) {
      k = function(a) {
        chrome.runtime.sendMessage({type:"sendData", val:{key:"m1", payload:[a]}}, function() {
        });
      };
      var z = document.location.href, A = !1;
      document.addEventListener("DOMContentLoaded", function(a) {
        if (!A) {
          try {
            if (z.startsWith("https://test.keepa.com") || z.startsWith("https://keepa.com")) {
              var b = document.createElement("div");
              b.id = "extension";
              b.setAttribute("type", F);
              b.setAttribute("version", D);
              document.body.appendChild(b);
              A = !0;
            }
          } catch (p) {
          }
        }
      });
      var H = !1;
      z.match(/^htt(p|ps):\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|nl|in|com\.mx|com\.br|com\.au)\/s\?/) ? (onlyStock = !0, f.onPageLoad()) : /((\/images)|(\/review)|(\/customer-reviews)|(ask\/questions)|(\/product-reviews))/.test(z) || /\/e\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/.test(z) || !(z.match(/^htt(p|ps):\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|nl|in|com\.mx|com\.br|com\.au)\/[^.]*?(\/|[?&]ASIN=)([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) || z.match(/^htt(p|ps):\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|nl|in|com\.mx|com\.br|com\.au)\/(.*?)\/dp\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))\//) || 
      z.match(/^htt(p|ps):\/\/.*?\.amzn\.(com).*?\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/)) ? z.match(/^htt(p|ps):\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|nl|es|in|com\.mx|com\.br|com\.au)\/[^.]*?\/(wishlist|registry)/) || z.match(/^htt(p|ps):\/\/w*?\.amzn\.(com)[^.]*?\/(wishlist|registry)/) || (z.match("^https?://.*?(?:seller).*?.amazon.(de|com|co.uk|co.jp|ca|fr|it|nl|es|in|com.mx|com.br|com.au)/") ? k("s" + f.getDomain(RegExp.$1)) : z.match(/^https?:\/\/.*?(?:af.?ilia|part|assoc).*?\.amazon\.(de|com|co\.uk|co\.jp|nl|ca|fr|it|es|in|com\.mx|com\.br|com\.au)\/home/) && 
      k("a" + f.getDomain(RegExp.$1))) : (f.onPageLoad(!1), H = !0);
      if (!C) {
        k = /^https?:\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|nl|in|com\.mx|com\.br|com\.au)\/(s([\/?])|gp\/bestsellers\/|gp\/search\/|.*?\/b\/)/;
        (H || z.match(k)) && document.addEventListener("DOMContentLoaded", function(a) {
          var b = null;
          chrome.runtime.sendMessage({type:"getFilters"}, function(a) {
            b = a;
            if (null != b && null != b.value) {
              var c = function() {
                var b = z.match("^https?://.*?.amazon.(de|com|co.uk|co.jp|ca|fr|it|es|in|com.br|nl|com.mx)/");
                if (H || b) {
                  var c = f.getDomain(RegExp.$1);
                  scanner.scan(a.value, function(a) {
                    a.key = "f1";
                    a.domainId = c;
                    chrome.runtime.sendMessage({type:"sendData", val:a}, function(a) {
                    });
                  });
                }
              };
              c();
              var g = document.location.href, e = -1, d = -1, k = -1;
              d = setInterval(function() {
                g != document.location.href && (g = document.location.href, clearTimeout(k), k = setTimeout(function() {
                  c();
                }, 2000), clearTimeout(e), e = setTimeout(function() {
                  clearInterval(d);
                }, 180000));
              }, 2000);
              e = setTimeout(function() {
                clearInterval(d);
              }, 180000);
            }
          });
        });
        k = document.location.href;
        k.match("^https?://.*?.amazon.(de|com|co.uk|co.jp|ca|fr|it|es|in|nl|com.mx|com.br|com.au)/") && -1 == k.indexOf("aws.amazon.") && -1 == k.indexOf("music.amazon.") && -1 == k.indexOf("services.amazon.") && -1 == k.indexOf("primenow.amazon.") && -1 == k.indexOf("kindle.amazon.") && -1 == k.indexOf("watch.amazon.") && -1 == k.indexOf("developer.amazon.") && -1 == k.indexOf("skills-store.amazon.") && -1 == k.indexOf("pay.amazon.") && document.addEventListener("DOMContentLoaded", function(a) {
          setTimeout(function() {
            chrome.runtime.onMessage.addListener(function(a, c, h) {
              switch(a.key) {
                case "collectASINs":
                  a = {};
                  var b = !1;
                  c = (document.querySelector("#main") || document.querySelector("#zg") || document.querySelector("#pageContent") || document.querySelector("#wishlist-page") || document.querySelector("#merchandised-content") || document.querySelector("#reactApp") || document.querySelector("[id^='contentGrid']") || document.querySelector("#container") || document.querySelector(".a-container") || document).getElementsByTagName("a");
                  if (void 0 != c && null != c) {
                    for (var e = 0; e < c.length; e++) {
                      var d = c[e].href;
                      /\/images/.test(d) || /\/e\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/.test(d) || !d.match(/^https?:\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|nl|in|com\.mx|com\.br|com\.au)\/[^.]*?(?:\/|\?ASIN=)([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) && !d.match(/^https?:\/\/.*?\.amzn\.(com)[^.]*?\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) || (b = RegExp.$2, d = f.getDomain(RegExp.$1), "undefined" === typeof a[d] && (a[d] = []), a[d].includes(b) || a[d].push(b), b = !0);
                    }
                  }
                  if (b) {
                    h(a);
                  } else {
                    return alert("Keepa: No product ASINs found on this page."), !1;
                  }
                  break;
                default:
                  h({});
              }
            });
            chrome.storage.local.get(["overlayPriceGraph", "webGraphType", "webGraphRange"], function(a) {
              "undefined" == typeof a && (a = {});
              try {
                var b = a.overlayPriceGraph, c = a.webGraphType;
                try {
                  c = JSON.parse(c);
                } catch (r) {
                }
                var f = a.webGraphRange;
                try {
                  f = Number(f);
                } catch (r) {
                }
                var e;
                if (1 == b) {
                  var d = document.getElementsByTagName("a"), k = 0 < document.location.href.indexOf("/offer-listing/");
                  if (void 0 != d && null != d) {
                    for (e = 0; e < d.length; e++) {
                      var l = d[e].href;
                      /\/images/.test(l) || /\/e\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/.test(l) || !l.match(/^https?:\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|in|com\.mx)\/[^.]*?(?:\/|\?ASIN=)([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) && !l.match(/^https?:\/\/.*?\.amzn\.(com)[^.]*?\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) || (k || -1 == l.indexOf("offer-listing")) && y.add_events(c, f, d[e], l, RegExp.$1, RegExp.$2);
                    }
                  }
                  var m = function(a) {
                    if ("A" == a.nodeName) {
                      var b = a.href;
                      /\/images/.test(b) || /\/e\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/.test(b) || !b.match(/^https?:\/\/.*?\.amazon\.(de|com|co\.uk|co\.jp|ca|fr|it|es|in|com\.mx)\/[^.]*?(?:\/|\?ASIN=)([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) && !b.match(/^https?:\/\/.*?\.amzn\.(com)[^.]*?\/([BC][A-Z0-9]{9}|\d{9}(!?X|\d))/) || (k || -1 == b.indexOf("offer-listing")) && y.add_events(c, f, a, b, RegExp.$1, RegExp.$2);
                    }
                  }, n = new MutationObserver(function(a) {
                    a.forEach(function(a) {
                      try {
                        if ("childList" === a.type) {
                          for (e = 0; e < a.addedNodes.length; e++) {
                            m(a.addedNodes[e]);
                            for (var b = a.addedNodes[e].children; null != b && "undefined" != b && 0 < b.length;) {
                              for (var c = [], d = 0; d < b.length; d++) {
                                m(b[d]);
                                try {
                                  if (b[d].children && 0 < b[d].children.length) {
                                    for (var f = 0; f < b[d].children.length && 30 > f; f++) {
                                      c.push(b[d].children[f]);
                                    }
                                  }
                                } catch (G) {
                                }
                              }
                              b = c;
                            }
                          }
                        } else {
                          if (c = a.target.getElementsByTagName("a"), "undefined" != c && null != c) {
                            for (b = 0; b < c.length; b++) {
                              m(c[b]);
                            }
                          }
                        }
                        m(a.target);
                      } catch (G) {
                      }
                    });
                  });
                  n.observe(document.querySelector("html"), {childList:!0, attributes:!1, characterData:!1, subtree:!0, attributeOldValue:!1, characterDataOldValue:!1});
                  window.onunload = function I() {
                    try {
                      window.detachEvent("onunload", I), n.disconnect();
                    } catch (L) {
                    }
                  };
                }
              } catch (r) {
              }
            });
          }, 100);
        });
        var y = {image_urls_main:[], pf_preview_current:"", preview_images:[], tld:"", createNewImageElement:function(a, b, c) {
          a = a.createElement("img");
          a.style.borderTop = "2px solid #ff9f29";
          a.style.borderBottom = "3px solid grey";
          a.style.display = "block";
          a.style.position = "relative";
          a.style.padding = "5px";
          a.style.width = b + "px";
          a.style.height = c + "px";
          a.style.maxWidth = b + "px";
          a.style.maxHeight = c + "px";
          return a;
        }, preview_image:function(a, b, c, f, g, e) {
          try {
            var d = c.originalTarget.ownerDocument;
          } catch (r) {
            d = document;
          }
          if (!d.getElementById("pf_preview")) {
            var h = d.createElement("div");
            h.id = "pf_preview";
            h.addEventListener("mouseout", function(a) {
              y.clear_image(a);
            }, !1);
            h.style.boxShadow = "rgb(68, 68, 68) 0px 1px 7px -2px";
            h.style.position = "fixed";
            h.style.zIndex = "10000000";
            h.style.bottom = "0px";
            h.style.right = "0px";
            h.style.margin = "12px 12px";
            h.style.backgroundColor = "#fff";
            d.body.appendChild(h);
          }
          y.pf_preview_current = d.getElementById("pf_preview");
          if (!y.pf_preview_current.firstChild) {
            h = Math.max(Math.floor(0.3 * d.defaultView.innerHeight), 128);
            var k = Math.max(Math.floor(0.3 * d.defaultView.innerWidth), 128), l = 2;
            if (300 > k || 150 > h) {
              l = 1;
            }
            1000 < k && (k = 1000);
            1000 < h && (h = 1000);
            y.pf_preview_current.current = -1;
            y.pf_preview_current.a = g;
            y.pf_preview_current.href = f;
            y.pf_preview_current.size = Math.floor(1.1 * Math.min(k, h));
            d.defaultView.innerWidth - c.clientX < 1.05 * k && d.defaultView.innerHeight - c.clientY < 1.05 * h && (c = d.getElementById("pf_preview"), c.style.right = "", c.style.left = "6px");
            g = "https://graph.keepa.com/pricehistory.png?type=" + l + "&asin=" + g + "&domain=" + e + "&width=" + k + "&height=" + h;
            g = "undefined" == typeof a ? g + "&amazon=1&new=1&used=1&salesrank=1&range=365" : g + ("&amazon=" + a[0] + "&new=" + a[1] + "&used=" + a[2] + "&salesrank=" + a[3] + "&range=" + b + "&fba=" + a[10] + "&fbm=" + a[7] + "&bb=" + a[18] + "&ld=" + a[8] + "&bbu=" + a[32] + "&pe=" + a[33] + "&wd=" + a[9]);
            d.getElementById("pf_preview").style.display = "block";
            var m = y.createNewImageElement(d, k, h);
            y.pf_preview_current.appendChild(m);
            fetch(g).then(function(a) {
              try {
                if ("FAIL" === a.headers.get("screenshot-status")) {
                  return null;
                }
              } catch (I) {
              }
              return a.blob();
            }).then(function(a) {
              null != a && m.setAttribute("src", URL.createObjectURL(a));
            });
          }
        }, clear_image:function(a) {
          try {
            try {
              var b = a.originalTarget.ownerDocument;
            } catch (h) {
              b = document;
            }
            var c = b.getElementById("pf_preview");
            c.style.display = "none";
            c.style.right = "2px";
            c.style.left = "";
            y.pf_preview_current.innerHTML = "";
          } catch (h) {
          }
        }, add_events:function(a, b, c, f, g, e) {
          0 <= f.indexOf("#") || (y.tld = g, "pf_prevImg" != c.getAttribute("keepaPreview") && (c.addEventListener("mouseover", function(c) {
            y.preview_image(a, b, c, f, e, g);
            return !0;
          }, !0), c.addEventListener("mouseout", function(a) {
            y.clear_image(a);
          }, !1), c.setAttribute("keepaPreview", "pf_prevImg")));
        }};
      }
    }
  }
})();

