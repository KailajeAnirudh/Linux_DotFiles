(self.webpackChunk=self.webpackChunk||[]).push([[637],{12989:(e,t,s)=>{s.d(t,{G:()=>r});var i=s(23239),n=s(5114),o=s(624);class r extends o.J{constructor(){super(),this._popoverActionsHandler=i.h.create(n.none)}setPopoverActionsHandler(e){this._popoverActionsHandler.set(n.some(e))}get popoverActionsHandlerAtom(){return this._popoverActionsHandler}}},517:(e,t,s)=>{s.d(t,{z:()=>o});var i=s(5114),n=s(23239);class o{constructor(){this.activePopoverStack=n.h.create(i.none),this.activePopoverView=n.h.create(i.none),this._popoverActionsHandler=n.h.create(i.none)}addPopover(e,t){}getInteractionActions(e,t,s){return[]}removePopover(e,t){}switchView(e){}get popoverActionsHandlerAtom(){return this._popoverActionsHandler}setPopoverActionsHandler(e){}}},65586:(e,t,s)=>{s.r(t),s.d(t,{createSduiBasicEngine:()=>w,default:()=>f});var i=s(5114),n=s(21038),o=s(24055),r=s(12054),a=s(12052),d=s(31903),c=s(9922),p=s(85985),l=s(77176),u=s(80358),h=s(84966),v=s(57050),m=s(86782),A=s(12989),g=s(517);const w=({sduiBufferService:e,alertState:t,getByRawAlertId:s,sendUserAction:w,environment:f,capiClient:C,getGrammarlyGoAssistantMode:S,sduiPopoversSupported:b=!1})=>{const _=new c.w.Keeper,P=(0,v.ls)((e=>e.toString()),s,i.fromNullable,i.map((e=>n.j.AlertId.from(e.id)))),G=e.capiEvents.pipe(p.h(u.h.is("sdui_add","sdui_remove","sdui_update")),p.h(o.e.isSduiEvent),l.U(r.al.fromSource(r.i5.CAPI))),I=b?new A.G:new g.z,E=new a.G(G,P,I),x=new d.Q({sendUserAction:w}),y=new m.U(E,x,f,C,{openAssistant(e){throw new Error("Method is not implemented")},closePopover:t=>e.pushCapiSduiEvent({kind:"sdui_remove",sduiRootId:h.t_.Id.create(t)})},S);return I.setPopoverActionsHandler(y),_.push(t.subscribe((e=>E.notifyAlertsChanged())),e.capiEvents.pipe(p.h(u.h.is("session_started")),p.h((e=>e.isNewSession))).subscribe((e=>E.onSessionStarted())),e.capiEvents.pipe(p.h(u.h.is("finish")),p.h((e=>0===e.revision))).subscribe((e=>E.onFirstCheckingFinished())),E),{sduiBufferService:e,sduiManager:E,sduiFeedbackService:x,dispose:()=>{_.dispose()}}},f={createSduiBasicEngine:w}},86782:(e,t,s)=>{s.d(t,{U:()=>F});var i=s(23239),n=s(72812),o=s(14454),r=s(57757),a=s(5114),d=s(31668),c=s(22232),p=s(71249),l=s(73975),u=s(57050),h=s(40151),v=s(95195),m=s(8125);class A{constructor(){this.feed=h.E,this.currentFeed=a.none,this.onFeedRemove=h.E,this.onFeedEmpty=h.E,this.header=h.E,this.footer=h.E,this.pushFeed=()=>v.right(null),this.popFeed=m.Q1,this.focusCard=()=>v.right(null),this.notifyCardFocused=m.Q1,this.dispose=m.Q1}}var g,w=s(33194),f=s(39040),C=s(32260),S=s(8901),b=s(32952),_=s(9922),P=s(41398),G=s(77176),I=s(93508),E=s(19751),x=s(69627),y=s(31528),M=s(26328),k=s(66268);class F{constructor(e,t,s,o,r,d){this.sduiManager=e,this.sduiFeedbackService=t,this.environment=s,this.capiClient=o,this.integrationModel=r,this.getGrammarlyGoAssistantMode=d,this._focusedItem=i.h.create(a.none),this._sduiInlineCardActions=new b.xQ,this._subs=new _.w.Keeper,this.focusCard=e=>{this._focusedItem.set(a.some(e))},this.handleSduiCardAction=(e,t,s)=>{this._sduiInlineCardActions.next({cardAction:e,cardModel:t,match:s})},this.handleGButtonPopoverAction=e=>{this._sduiViewModel.sduiActionEvents.next(e),(0,u.zG)(F.getSduiFeedActions(e.actions),a.map((t=>this.integrationModel.openAssistant({type:y.WT.sdui,action:{...e,actions:t,type:k.lY.Type.sduiCardAction}}))))},this.dispose=()=>this._subs.dispose(),this._sduiViewModel=new n.x(t,e,new A,o,this._focusedItem.view(a.map((e=>e.id))));const c=this._sduiViewModel.sduiActionEvents.pipe(P.M(this._sduiInlineCardActions.pipe(G.U(a.some),I.O(a.none)))).subscribe((([e,t])=>this._nativeHandling(e,(0,u.zG)(t,a.map((e=>e.match))),(0,u.zG)(t,a.map((e=>e.cardModel)))))),p=this._sduiInlineCardActions.pipe(E.skipBy(g.eq)).subscribe((({cardAction:e})=>this._sduiViewModel.sduiActionEvents.next(e)));this._subs.push(c,p,this._sduiViewModel)}_nativeHandling({actions:e,cardId:t,sourceId:s,scope:i},n,o){(0,u.zG)(F.getSduiFeedActions(e),a.map((e=>{(0,u.zG)(o,a.map((n=>{n.onOpenExpandedViewBySduiAction(w.Oe.create(e,t,s,i))})))}))),e.forEach((e=>{switch(e.type){case"nextCard":case"prevCard":case"openSettings":case"openToneDetector":case"openFeedback":case"openLearnMore":case"transition":case"openCreateSnippetModal":case"nativeOpenAssistant":case"selectAlternative":case"highlightAlert":case"openPerformanceScore":case"nativeOpenUserSatisfactionFeedback":case"enablePlagiarismCheck":case"disablePlagiarismCheck":case"showHighlights":case"hideHighlights":case"notify":case"switchView":case"newRevision":case"interactPopover":case"enableWritingExpertCheck":case"disableWritingExpertCheck":case"pushAssistantFeed":case"popAssistantFeed":case"focusAssistantCard":return;case"closePopover":return void this.integrationModel.closePopover(e.rootPopoverId);case"openLink":return void this.environment.actions.openPopup(new d.Z(e.url));case"copyToClipboard":return void(0,r.vQ)(e.text);case"stopApplyingAlerts":case"upgradeToPremium":return(0,u.zG)(o,a.map((e=>e.openPlanComparisonPage({utmCampaign:S.L.Place.assistantCardList})))),void(0,u.zG)(n,a.map((e=>e.hide())));case"applyAlerts":return void(0,u.zG)(n,a.map((t=>{null!=t.alert&&((0,u.zG)((0,f.UQ)(t.alert,t.plainText),(s=>new C.U_(s,(s=>t.replace(null!=s?s:"",!1,e.alternativeIndex)))),(t=>(0,u.zG)(a.fromNullable(t.replacements[e.alternativeIndex]),a.map((e=>t.getOnReplace(e)()))))),t.hide())})));case"closeCard":case"removeRoot":return void(0,u.zG)(n,a.map((e=>e.hide())));case"removeAlerts":return void(0,u.zG)(n,a.map((e=>{e.ignore(),e.hide()})));case"openGrammarlyGo":return void(0,u.zG)(a.sequenceArray([o,n]),a.map((([e,t])=>{var s,i;const n=(t,s)=>{const i=this.getGrammarlyGoAssistantMode();e.onOpenGrammarlyGo({assistantMode:i,genAIParams:t,source:s})};if(t.alert&&(0,x.t)(t.alert.patternName)){const e=(0,f.UQ)(t.alert,t.plainText);if(e.length>0&&e[0].newText){const s=e[0].newText;n({genAISessionMode:"pushRewrite",writingExpertContext:{alertId:t.alert.id,alertPname:t.alert.patternName,originalText:t.alert.text,replacementText:s},transformRange:t.alert.transformRange},{kind:"inlinePushRewriteSource"})}else n({genAISessionMode:"ideation"},{kind:"inlinePushRewriteSource"})}else n({genAISessionMode:"ideation"},{kind:"inlineCardSource",alertId:null!==(i=null===(s=t.alert)||void 0===s?void 0:s.id)&&void 0!==i?i:null})})));default:(0,c.L0)(e)}}))}}F.getSduiFeedActions=(0,u.ls)(p.hX((e=>"popAssistantFeed"===e.type||"pushAssistantFeed"===e.type||"focusAssistantCard"===e.type)),M.nI),function(e){e.eq=l.MW({cardAction:o.t.eq})}(g||(g={}))},57757:(e,t,s)=>{s.d(t,{Sz:()=>o,vQ:()=>n});var i=s(95195);async function n(e,t=self){if(function(e=self){var t,s;return!!(null===(s=null===(t=e.navigator)||void 0===t?void 0:t.clipboard)||void 0===s?void 0:s.writeText)}(t))return t.navigator.clipboard.writeText(e);throw new Error("Clipboard API not supported")}function o(e,t=self){return n(e,t).then((()=>i.right(void 0))).catch((e=>i.left(e instanceof Error?e:new Error(String(e)))))}}}]);