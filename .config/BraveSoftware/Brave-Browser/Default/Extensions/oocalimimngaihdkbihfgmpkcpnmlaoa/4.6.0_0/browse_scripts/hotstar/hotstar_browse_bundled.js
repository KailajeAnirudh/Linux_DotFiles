/*******************************************************
* Copyright (C) 2018-2024 WP Interactive Media, Inc. - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
*******************************************************/
(()=>{var t={5640:()=>{!function(t,n,e){"use strict";if(t.MutationObserver&&"undefined"!=typeof HTMLElement){var i,r=0,o=(i=HTMLElement.prototype.matches||HTMLElement.prototype.webkitMatchesSelector||HTMLElement.prototype.mozMatchesSelector||HTMLElement.prototype.msMatchesSelector,{matchesSelector:function(t,n){return t instanceof HTMLElement&&i.call(t,n)},addMethod:function(t,n,e){var i=t[n];t[n]=function(){return e.length==arguments.length?e.apply(this,arguments):"function"==typeof i?i.apply(this,arguments):void 0}},callCallbacks:function(t,n){n&&n.options.onceOnly&&1==n.firedElems.length&&(t=[t[0]]);for(var e,i=0;e=t[i];i++)e&&e.callback&&e.callback.call(e.elem,e.elem);n&&n.options.onceOnly&&1==n.firedElems.length&&n.me.unbindEventWithSelectorAndCallback.call(n.target,n.selector,n.callback)},checkChildNodesRecursively:function(t,n,e,i){for(var r,s=0;r=t[s];s++)e(r,n,i)&&i.push({callback:n.callback,elem:r}),r.childNodes.length>0&&o.checkChildNodesRecursively(r.childNodes,n,e,i)},mergeArrays:function(t,n){var e,i={};for(e in t)t.hasOwnProperty(e)&&(i[e]=t[e]);for(e in n)n.hasOwnProperty(e)&&(i[e]=n[e]);return i},toElementsArray:function(n){return void 0===n||"number"==typeof n.length&&n!==t||(n=[n]),n}}),s=function(){var t=function(){this.t=[],this.i=null,this.o=null};return t.prototype.addEvent=function(t,n,e,i){var r={target:t,selector:n,options:e,callback:i,firedElems:[]};return this.i&&this.i(r),this.t.push(r),r},t.prototype.removeEvent=function(t){for(var n,e=this.t.length-1;n=this.t[e];e--)if(t(n)){this.o&&this.o(n);var i=this.t.splice(e,1);i&&i.length&&(i[0].callback=null)}},t.prototype.beforeAdding=function(t){this.i=t},t.prototype.beforeRemoving=function(t){this.o=t},t}(),u=function(n,i){var r=new s,u=this,c={fireOnAttributesModification:!1};return r.beforeAdding((function(e){var r,o=e.target;o!==t.document&&o!==t||(o=document.getElementsByTagName("html")[0]),r=new MutationObserver((function(t){i.call(this,t,e)}));var s=n(e.options);r.observe(o,s),e.observer=r,e.me=u})),r.beforeRemoving((function(t){t.observer.disconnect()})),this.bindEvent=function(t,n,e){n=o.mergeArrays(c,n);for(var i=o.toElementsArray(this),s=0;s<i.length;s++)r.addEvent(i[s],t,n,e)},this.unbindEvent=function(){var t=o.toElementsArray(this);r.removeEvent((function(n){for(var i=0;i<t.length;i++)if(this===e||n.target===t[i])return!0;return!1}))},this.unbindEventWithSelectorOrCallback=function(t){var n,i=o.toElementsArray(this),s=t;n="function"==typeof t?function(t){for(var n=0;n<i.length;n++)if((this===e||t.target===i[n])&&t.callback===s)return!0;return!1}:function(n){for(var r=0;r<i.length;r++)if((this===e||n.target===i[r])&&n.selector===t)return!0;return!1},r.removeEvent(n)},this.unbindEventWithSelectorAndCallback=function(t,n){var i=o.toElementsArray(this);r.removeEvent((function(r){for(var o=0;o<i.length;o++)if((this===e||r.target===i[o])&&r.selector===t&&r.callback===n)return!0;return!1}))},this},c=new function(){var t={fireOnAttributesModification:!1,onceOnly:!1,existing:!1};function n(t,n,i){return!(!o.matchesSelector(t,n.selector)||(t.u===e&&(t.u=r++),-1!=n.firedElems.indexOf(t.u)))&&(n.firedElems.push(t.u),!0)}var i=(c=new u((function(t){var n={attributes:!1,childList:!0,subtree:!0};return t.fireOnAttributesModification&&(n.attributes=!0),n}),(function(t,e){t.forEach((function(t){var i=t.addedNodes,r=t.target,s=[];null!==i&&i.length>0?o.checkChildNodesRecursively(i,e,n,s):"attributes"===t.type&&n(r,e,s)&&s.push({callback:e.callback,elem:r}),o.callCallbacks(s,e)}))}))).bindEvent;return c.bindEvent=function(n,e,r){void 0===r?(r=e,e=t):e=o.mergeArrays(t,e);var s=o.toElementsArray(this);if(e.existing){for(var u=[],c=0;c<s.length;c++)for(var a=s[c].querySelectorAll(n),f=0;f<a.length;f++)u.push({callback:r,elem:a[f]});if(e.onceOnly&&u.length)return r.call(u[0].elem,u[0].elem);setTimeout(o.callCallbacks,1,u)}i.call(this,n,e,r)},c},a=new function(){var t={};function n(t,n){return o.matchesSelector(t,n.selector)}var e=(a=new u((function(){return{childList:!0,subtree:!0}}),(function(t,e){t.forEach((function(t){var i=t.removedNodes,r=[];null!==i&&i.length>0&&o.checkChildNodesRecursively(i,e,n,r),o.callCallbacks(r,e)}))}))).bindEvent;return a.bindEvent=function(n,i,r){void 0===r?(r=i,i=t):i=o.mergeArrays(t,i),e.call(this,n,i,r)},a};n&&l(n.fn),l(HTMLElement.prototype),l(NodeList.prototype),l(HTMLCollection.prototype),l(HTMLDocument.prototype),l(Window.prototype);var f={};return h(c,f,"unbindAllArrive"),h(a,f,"unbindAllLeave"),f}function h(t,n,e){o.addMethod(n,e,t.unbindEvent),o.addMethod(n,e,t.unbindEventWithSelectorOrCallback),o.addMethod(n,e,t.unbindEventWithSelectorAndCallback)}function l(t){t.arrive=c.bindEvent,h(c,t,"unbindArrive"),t.leave=a.bindEvent,h(a,t,"unbindLeave")}}(window,"undefined"==typeof jQuery?null:jQuery,void 0)}},n={};function e(i){var r=n[i];if(void 0!==r)return r.exports;var o=n[i]={exports:{}};return t[i](o,o.exports,e),o.exports}(()=>{"use strict";var t;!function(t){t.NETFLIX="Netflix",t.HULU="Hulu",t.DISNEY_PLUS="Disney",t.DISNEY_PLUS_MENA="DisneyMena",t.HBO_MAX="HBOMax",t.MAX="Max",t.YOUTUBE="Youtube",t.YOUTUBE_TV="YoutubeTV",t.AMAZON="Amazon",t.CRUNCHYROLL="Crunchyroll",t.ESPN="ESPN+",t.PARAMOUNT="Paramount+",t.FUNIMATION="Funimation",t.HOTSTAR="Hotstar",t.PEACOCK="peacock",t.STAR_PLUS="Starplus",t.PLUTO_TV="PlutoTV",t.APPLE_TV="AppleTV",t.JIO_CINEMA="JioCinema",t.TUBI_TV="TubiTV",t.MUBI="Mubi",t.STAN="Stan",t.CRAVE="Crave",t.SLING="slingtv",t.FUBO="fubo",t.PHILO="philo",t.VIKI="viki",t.SPOTIFY="spotify",t.NATIVE_VIKI="native_party_Viki"}(t||(t={}));var n;!function(t){t.REGISTER="register",t.PARTY_START="party_start",t.PARTY_JOIN="party_join",t.PARTY_END="party_end",t.PARTY_SHARE="party_share"}(n||(n={}));const i=chrome.runtime.id,r=["crunchyroll","funimation","paramount","peacock","hotstar","starplus","espn","appletv","jiocinema","mubi","stan","crave","sling","philo","fubo","viki"];var o=console.log.bind(window.console),s=function(t,n,e,i){return new(e||(e=Promise))((function(r,o){function s(t){try{c(i.next(t))}catch(t){o(t)}}function u(t){try{c(i.throw(t))}catch(t){o(t)}}function c(t){var n;t.done?r(t.value):(n=t.value,n instanceof e?n:new e((function(t){t(n)}))).then(s,u)}c((i=i.apply(t,n||[])).next())}))};const u=new class{addListener(t){chrome.runtime.onMessage.addListener(t),chrome.runtime.onMessage.addListener(t)}removeListener(t){chrome.runtime.onMessage.removeListener(t)}sendMessageToTabAsync(t,n,e=2e4){return s(this,void 0,void 0,(function*(){return new Promise(((i,r)=>{const s=setTimeout((()=>{console.log("send timeout"),r("Message Timeout")}),e);try{chrome.tabs.sendMessage(n,t,(n=>{chrome.runtime.lastError&&o(chrome.runtime.lastError.message+JSON.stringify(t)),clearTimeout(s),i(n)}))}catch(t){clearTimeout(s),r(t)}}))}))}h(t,n){return s(this,void 0,void 0,(function*(){return new Promise(((e,r)=>{let o=null;n&&(o=setTimeout((()=>{r({error:"Send Message Timeout"})}),n));try{chrome.runtime.sendMessage(i,t,(n=>{chrome.runtime.lastError&&console.log(chrome.runtime.lastError.message+JSON.stringify(t)),o&&clearTimeout(o),e(n)}))}catch(t){o&&clearTimeout(o),r(t)}}))}))}};class c{constructor(){this.l=this.v.bind(this),this.m=[],this.p()}addMessageListener(t){this.m.push(t)}removeMessageListener(t){this.m=this.m.filter((t=>{}))}p(){u.addListener(this.l)}teardown(){this.m=[],u.removeListener(this.l)}v(t,n,e){if(!this.g(t))return!1;return!!this.T(t,n,e)||(e({}),!1)}g(t){return"Content_Script"===t.target}T(t,n,e){let i=!1;return this.m.forEach((r=>{r.onMessage(t,n,e)&&(i=!0)})),i}}var a;e(5640);!function(t){t.CREATE_SESSION="createSession",t.RE_INJECT="reInject",t.GET_INIT_DATA="getInitData",t.GET_INIT_USER_SETTINGS="getInitUserSettings",t.IS_CONTENT_SCRIPT_READY="isContentScriptReady",t.SET_CHAT_VISIBLE="setChatVisible",t.DISCONNECT="teardown",t.CLOSE_POPUP="closePopup",t.GET_SCHEDULED_EVENTS="getScheduledEvents",t.GET_RECENT_SCHEDULED_EVENTS="getRecentScheduledEvents",t.GOOGLE_SIGN_IN="googleSignIn",t.EMAIL_SIGN_IN="emailSignIn",t.FORGOT_PASSWORD="forgotPassword",t.EMAIL_SIGN_UP="emailSignUp",t.ON_GOOGLE_SIGN_IN="OnGoogleSignIn",t.CREATE_SCHEDULED_EVENT="createScheduledEvent",t.DELETE_SCHEDULED_EVENT="deleteScheduledEvent",t.SET_USER_STATUS="SET_USER_STATUS",t.SIGN_IN_CREATE="SIGN_IN_CREATE",t.REDIRECT_DATA_SET="redirectDataSet"}(a||(a={}));class f extends class extends class{constructor(t,n,e){this.sender=t,this.target=n,this.type=e}}{constructor(t,n,e){super(t,n,e),this.type=e}}{constructor(t,n){super(t,n,a.GET_INIT_USER_SETTINGS)}}var h,l=function(t,n,e,i){return new(e||(e=Promise))((function(r,o){function s(t){try{c(i.next(t))}catch(t){o(t)}}function u(t){try{c(i.throw(t))}catch(t){o(t)}}function c(t){var n;t.done?r(t.value):(n=t.value,n instanceof e?n:new e((function(t){t(n)}))).then(s,u)}c((i=i.apply(t,n||[])).next())}))};class d{constructor(t,n){this.l=new c,this.l.addMessageListener(this),this.serviceName=t,this.colloquialServiceName=n,this.M(),this.S(),this._()}isLoggingIn(){return!1}S(){return l(this,void 0,void 0,(function*(){const t=yield u.h(new f("Popup","Service_Background"));r.includes(this.serviceName)&&!(null==t?void 0:t.has_premium)||this.L()}))}L(){!function(t){const n=document.createElement("script");n.setAttribute("tpInjected",""),n.src=t,(document.head||document.documentElement).appendChild(n),n.remove()}(chrome.extension.getURL(`browse_scripts/${this.serviceName}/${this.serviceName}_browse_injected_bundled.js`))}M(){chrome.runtime.connect().onMessage.addListener((()=>{console.log("Ping received")}))}_(){return l(this,void 0,void 0,(function*(){}))}onMessage(t,n,e){return!0}}window.teleparty&&(null===(h=window.teleparty)||void 0===h?void 0:h.browseScriptInjected)||(window.teleparty||(window.teleparty={}),window.teleparty.browseScriptInjected=!0,new d("hotstar",t.HOTSTAR))})()})();