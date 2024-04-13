!function(){"use strict";function e(t){return e="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e},e(t)}function t(t){var i=function(t,i){if("object"!==e(t)||null===t)return t;var o=t[Symbol.toPrimitive];if(void 0!==o){var s=o.call(t,i||"default");if("object"!==e(s))return s;throw new TypeError("@@toPrimitive must return a primitive value.")}return("string"===i?String:Number)(t)}(t,"string");return"symbol"===e(i)?i:String(i)}function i(e,i,o){return(i=t(i))in e?Object.defineProperty(e,i,{value:o,enumerable:!0,configurable:!0,writable:!0}):e[i]=o,e}let o=function(e){return e.Opal="Opal",e.Extension="Extension",e.SafariExtension="SafariExtension",e.SafariIOSExtension="SafariIOSExtension",e.Edge="Edge",e.EdgeMobile="EdgeMobile",e.Sapphire="Sapphire",e.RBC="RBC",e.EdgeAndroid="EdgeAndroid",e.EdgeiOS="EdgeiOS",e.EdgeDiscover="EdgeDiscover",e}({});const s=[o.EdgeMobile,o.EdgeAndroid,o.EdgeiOS];o.Edge;new Set(["amazon.com","amazon.ca","amazon.co.uk","amazon.co.jp","alibaba.com"]),new Map(Object.entries({"etsy.com":"receipt_id","target.com":"referenceId","tmall.com":"bizOrderId"}));const r="COMPONENT_TO_FOCUS_IN_SHORELINE";Object.keys({"bestbuy.com":{policyDays:15,supportPageUrl:"https://www.bestbuy.com/site/help-topics/price-match-guarantee/pcmcat290300050002.c?id=pcmcat290300050002"},"costco.com":{policyDays:30,supportPageUrl:"https://customerservice.costco.com/app/answers/detail/a_id/628/~/price-adjustment---costco.com-orders",useCartAtPathname:"/checkoutcartdisplayview"},"kohls.com":{policyDays:14,supportPageUrl:"https://cs.kohls.com/app/answers/detail/a_id/90/~/price-match-policy"},"target.com":{policyDays:14,supportPageUrl:"https://help.target.com/help/subcategoryarticle?childcat=Price+Match+Guarantee&parentcat=Policies+%26+Guidelines&searchQuery=search+help",useCartAtPathname:"/cart"},"dickssportinggoods.com":{policyDays:14,supportPageUrl:"https://www.dickssportinggoods.com/s/price-match-policy",useCartAtPathname:"/orderitemdisplay"},"jcpenney.com":{policyDays:14,supportPageUrl:"https://www.jcpenney.com/m/customer-service/our-lowest-price-guarantee"},"macys.com":{policyDays:10,supportPageUrl:"https://customerservice-macys.com/articles/how-can-i-get-a-price-adjustment",useCartAtPathname:"/my-bag",hasCsrError:!0},"ashleyfurniture.com":{policyDays:30,supportPageUrl:"https://www.ashleyfurniture.com/price-match/"},"gap.com":{policyDays:14,supportPageUrl:"https://www.gap.com/customerService/info.do?cid=1192378"},"staples.com":{policyDays:14,supportPageUrl:"https://www.staples.com/sbd/cre/marketing/pmg/index.html"}});let a=null;const n="test-shopping-localstorage";function c(e){let t=null;return l()&&(t=window.localStorage.getItem(e)),t}function l(){try{if(null!==a)return a;"undefined"!=typeof window&&window?.localStorage&&(window.localStorage.setItem(n,n),window.localStorage.getItem(n),window.localStorage.removeItem(n),a=!0)}catch(e){a=!1}return a}class u{static Sleep(e){return new Promise((t=>setTimeout(t,e)))}static StringifyMap(e,t){return t instanceof Map?{dataType:"Map",value:Array.from(t.entries())}:t}static ParseMap(e,t){return"object"==typeof t&&null!==t&&"Map"===t.dataType?new Map(t.value):t}static async WaitForCondition(e,t,i){const o=(new Date).getTime();for(;!await e()&&o+t>(new Date).getTime();)await u.Sleep(i??100);return await e()}static async WaitUntilCondition(e,t){const i=(new Date).getTime();for(;i+t>(new Date).getTime();){if(await e())return!0;await u.Sleep(100)}return!1}static async WaitForSyncCondition(e,t){const i=(new Date).getTime();for(;i+t>(new Date).getTime();){if(e())return!0;await u.Sleep(100)}return!1}static IsValidDataField(e){return null!=e&&e.length>0&&"null"!==e}static IsPageMatch(e,t,i,o){let s=!1;if(u.IsValidDataField(e)&&(s=u.IsOnPage(e,i)),u.IsValidDataField(t))try{!o&&location.href?.toLocaleLowerCase()?.includes(i.toLocaleLowerCase())&&"chrome-untrusted://shopping/"!==location.href&&(o=location.href?.toLocaleLowerCase()),s=u.IsPageRegexMatch(t,o??i)}catch{}return s}static IsPageRegexMatch(e,t){if(u.IsValidDataField(e)){return new RegExp(e).test(t.toLowerCase())}return!1}static IsOnPage(e,t){if(u.IsValidDataField(e)){const i=e.toLowerCase().replace(/\s+/g,"").split(","),o=t.toLowerCase();let s=!1;for(const e of i)if(o.indexOf(e)>=0){s=!0;break}return s}return!1}static ObserveUntil(e,t){const i=new MutationObserver((async()=>{e()&&(i.disconnect(),t())}));i.observe(document.body,{attributeFilter:["offsetWidth","offsetHeight"],childList:!0,subtree:!0})}static async MeasureExecutionTime(e,t){const i=performance.now();await e();return performance.now()-i}static DeepAssign(e,t){return Object.keys(t).forEach((i=>{"object"==typeof t[i]?(e[i]||Object.assign(e,{[i]:{}}),u.DeepAssign(e[i],t[i])):Object.assign(e,{[i]:t[i]})})),e}static scrollToModuleIfTargeted(e,t){c(r)===t&&setTimeout((()=>{e?.scrollIntoView({behavior:"smooth",block:"start"}),localStorage.removeItem(r)}),500)}}var p=u;var m=class{constructor(e,t,o,s,r,a,n,c,l,u,p,m,d){i(this,"Name",void 0),i(this,"Type",void 0),i(this,"Value",void 0),i(this,"IsMandatory",void 0),i(this,"Format",void 0),i(this,"WaitForVisible",void 0),i(this,"WaitForNotDisabled",void 0),i(this,"WaitBefore",void 0),i(this,"WaitAfter",void 0),i(this,"WaitForNotVisible",void 0),i(this,"NotAlwaysShown",void 0),i(this,"DynamicFetch",void 0),i(this,"ShouldValue",void 0),this.Name=e,this.Type=t,this.Value=o,this.IsMandatory=s,this.Format=r,this.WaitForVisible=a,this.WaitForNotDisabled=n,this.WaitBefore=c,this.WaitAfter=l,this.WaitForNotVisible=u,this.NotAlwaysShown=p,this.DynamicFetch=m,this.ShouldValue=d}};var d=class{constructor(e,t,o){if(i(this,"PageUrl",void 0),i(this,"Type",void 0),i(this,"CheckoutElements",void 0),this.PageUrl=e,this.Type=t,this.CheckoutElements=new Map,o)for(const e of o)if(e){const t=e.Name;let i=e.Value;t&&this.CheckoutElements.set(t,new m(t,e.Type,i,e.IsMandatory,e.Format,e.WaitForVisible,e.WaitForNotDisabled,e.WaitBefore,e.WaitAfter,e.WaitForNotVisble,e.NotAlwaysShown,e.DynamicFetch,e.ShouldValue))}}};class h{static Create(e){let t=JSON.parse(atob(e));const i=[],o=new Map,s=t?.[0]?.Group;if(s){let e=s;for(const i of t)if(i){const t=i.Group;if(t&&p.IsOnPage(i.PageUrl,location.pathname)){e=t;break}}t=t.map((t=>{if(t.Group===e)return t}))}for(const e of t)if(e){const t=e.Type;t&&!o.has(t)&&(i.push(t),o.set(t,new d(e.PageUrl,t,e.checkoutElements)))}return{map:o,array:i}}constructor(e){if(i(this,"DomainName",void 0),i(this,"AllcheckoutCompletionPages",void 0),i(this,"AllPageTypeArr",void 0),i(this,"AllCheckoutCompletionPagesStr",void 0),i(this,"IsExpressCheckoutEnabled",void 0),i(this,"CheckoutPageUrl",void 0),e){this.DomainName=e.domainName,this.CheckoutPageUrl=e.checkoutPageUrl,this.IsExpressCheckoutEnabled=e.isExpressCheckoutEnabled;const t=e.allCheckoutCompletionPagesStr;if(this.AllCheckoutCompletionPagesStr=t,t){const{map:e,array:i}=h.Create(t);this.AllcheckoutCompletionPages=e,this.AllPageTypeArr=i}}}}i(h,"PageTypeArr",[]);var f=h;let g=function(e){return e.CCNUpdate="CCNUpdate",e.CCName="CCName",e.CCFirstName="CCFirstName",e.CCMiddleName="CCMiddleName",e.CCLastName="CCLastName",e.CCExpiry="CCExpiry",e.CCExpiryMonth="CCExpiryMonth",e.CCExpiryYear="CCExpiryYear",e.CCSecurityCode="CCSecurityCode",e}({});class y{static HasVisibleElement(e){return y.CountVisibleElements(e)>0}static CountVisibleElements(e){if(!p.IsValidDataField(e))return 0;const t=e.split(";");for(const e of t){const t=y.CountVisibleElementsSingleSel(e);if(t>0)return t}return 0}static RunQuerySelectorAll(e,t){if(!p.IsValidDataField(e))return[];const i=e.split("<");let o;o=t?t.querySelectorAll(i[0]):document.querySelectorAll(i[0]);for(const e of i.slice(1)){const t=o[0]?.shadowRoot;if(!t)return[];o=t.querySelectorAll(e)}return o}static IsElementVisible(e){return e&&e.offsetWidth>0&&e.offsetHeight>0}static GetFirstVisibleElement(e,t){if(!p.IsValidDataField(e))return;const i=e.split(";");for(const e of i)try{const i=y.RunQuerySelectorAll(e,t);for(const e of i)if(y.IsElementVisible(e))return e}catch(e){}}static GetAllVisibleElements(e){if(!p.IsValidDataField(e))return[];const t=e.split(";"),i=[];for(const e of t){const t=y.RunQuerySelectorAll(e);for(const e of t)y.IsElementVisible(e)&&i.push(e)}return i}static GetTextValue(e,t){const i=e.split(";"),o=i[0],s=y.GetFirstVisibleElement(o,t);if(!s)return"";let r=s,a=r.innerText;if(1===i.length)r=y.NormalizeIfSuperscripted(s),a=r.innerText;else{const e=r.cloneNode(!0);let s=i[1];const n=y.GetFirstVisibleElement(s,r)??y.GetFirstVisibleElement(s,t);let c="";if(n&&n.innerText){if(c="."+n.innerText,r.contains(n)){const t=y.GetFirstMatchingElement(s,e);if(t?.innerText)e.removeChild(t);else{s.startsWith(o)&&(s=s.slice(o.length));const t=this.GetFirstMatchingElement(s,e);t?.innerText&&e.removeChild(t)}a=e?.innerText?e.innerText:a}a+=c}if(i.length>2){for(const t of i.slice(2)){const i=this.GetFirstMatchingElement(t,e);i?.innerText&&e.removeChild(i)}a=e?.innerText?e.innerText:a}a+=c}return y.StripInvalidJSONCharacters(a)}static GetItemizedData(e,t,i){let o="";if(e&&""!==e){const s=y.RunQuerySelectorAll(e,i);for(const e of s)e&&e.textContent&&(o+=e.textContent?.trim()+t)}return o}static StripInvalidJSONCharacters(e){return e.replace(/\n/gi,"")}static NormalizeIfSuperscripted(e){if(e&&e.innerHTML){if(e.innerHTML.toLowerCase().indexOf("</sup>")>-1)try{const t=e.cloneNode(!0),i=t.childNodes.length;for(let e=0;e<i;e++){const i=t.childNodes[e];if("SUP"===i.tagName){let e=i.innerText;const o=/[0-9\.]+/g.exec(e);if(null!==o)return e="."+o[0],i.innerText=e,t}}}catch(t){return e}}return e}static GetFirstMatchingElement(e,t){if(!p.IsValidDataField(e))return;const i=e.split(";");for(const e of i){const i=y.RunQuerySelectorAll(e,t);for(const e of i)if(e)return e}}static GetAllMatchingElements(e){if(!p.IsValidDataField(e))return[];const t=e.split(";"),i=[];for(const e of t)try{const t=y.RunQuerySelectorAll(e);for(const e of t)e&&i.push(e)}catch(e){}return i}static CountVisibleElementsSingleSel(e){if(!p.IsValidDataField(e))return 0;const t=y.RunQuerySelectorAll(e);let i=0;for(const e of t)y.IsElementVisible(e)&&i++;return i}}var C=y;function w(e,t){const i=document.createEvent("Events");i.initEvent("change",!0,!1);const o=document.createEvent("Events");o.initEvent("input",!0,!1);const s=new KeyboardEvent("keyup",{bubbles:!0,cancelable:!0,view:window}),r=C.GetFirstVisibleElement(e);if(!r)throw new Error("input box undefined");r.blur(),r.dispatchEvent(i),r.focus(),r.setAttribute("value",t),r.value=t,r.dispatchEvent(s),r.dispatchEvent(o),r.dispatchEvent(i)}window.RunIframeAction=function(e){let t="",i="";try{const o=JSON.parse(e[0]);t=o.Guid,i=o.ParentOrigin;const s=o.CommandName,r=o.Value,a=f.Create(o.AllCheckoutCompletionPagesStr)?.map,n=a.get("PaymentIframe");try{if(s===g.CCNUpdate){const e=n?.CheckoutElements.get("cardNumber");e&&w(e.Value,r)}else if(s===g.CCName){const e=n?.CheckoutElements.get("nameOnCard");e&&w(e.Value,r)}else if(s===g.CCFirstName){const e=n?.CheckoutElements.get("firstName");e&&w(e.Value,r)}else if(s===g.CCMiddleName){const e=n?.CheckoutElements.get("middleName");e&&w(e.Value,r)}else if(s===g.CCLastName){const e=n?.CheckoutElements.get("lastName");e&&w(e.Value,r)}else if(s===g.CCExpiry){const e=n?.CheckoutElements.get("expiry");e&&w(e.Value,r)}else if(s===g.CCExpiryMonth){const e=n?.CheckoutElements.get("expiryMonth");e&&w(e.Value,r)}else if(s===g.CCExpiryYear){const e=n?.CheckoutElements.get("expiryYear");e&&w(e.Value,r)}else if(s===g.CCSecurityCode){const e=n?.CheckoutElements.get("securityCode");e&&w(e.Value,r)}parent.postMessage({guid:t,status:"SUCCESS"},i)}catch(e){parent.postMessage({guid:t,status:"ERROR"},i)}}catch(e){parent.postMessage({guid:t,status:"ERROR"},i)}};const E=new class{initialize(e){e.splice(0,2),window.RunIframeAction(e)}};window.shoppingIframeRuntime=E}();