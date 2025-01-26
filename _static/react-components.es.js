import * as Q from "react";
import T, { isValidElement as xt, Children as fr, PureComponent as Xt, useMemo as FO, cloneElement as De, createElement as a0, useRef as o0, useState as Er, useEffect as Xf, useContext as qt, createContext as mr, Component as u0, forwardRef as zO } from "react";
import "react-dom";
var Ei = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
function Pe(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Ti = { exports: {} }, mn = {};
/**
 * @license React
 * react-jsx-runtime.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var ap;
function UO() {
  if (ap) return mn;
  ap = 1;
  var e = Symbol.for("react.transitional.element"), t = Symbol.for("react.fragment");
  function r(n, i, a) {
    var o = null;
    if (a !== void 0 && (o = "" + a), i.key !== void 0 && (o = "" + i.key), "key" in i) {
      a = {};
      for (var u in i)
        u !== "key" && (a[u] = i[u]);
    } else a = i;
    return i = a.ref, {
      $$typeof: e,
      type: n,
      key: o,
      ref: i !== void 0 ? i : null,
      props: a
    };
  }
  return mn.Fragment = t, mn.jsx = r, mn.jsxs = r, mn;
}
var gn = {};
/**
 * @license React
 * react-jsx-runtime.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var op;
function WO() {
  return op || (op = 1, process.env.NODE_ENV !== "production" && function() {
    function e(A) {
      if (A == null) return null;
      if (typeof A == "function")
        return A.$$typeof === U ? null : A.displayName || A.name || null;
      if (typeof A == "string") return A;
      switch (A) {
        case x:
          return "Fragment";
        case m:
          return "Portal";
        case P:
          return "Profiler";
        case _:
          return "StrictMode";
        case j:
          return "Suspense";
        case M:
          return "SuspenseList";
      }
      if (typeof A == "object")
        switch (typeof A.tag == "number" && console.error(
          "Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."
        ), A.$$typeof) {
          case I:
            return (A.displayName || "Context") + ".Provider";
          case E:
            return (A._context.displayName || "Context") + ".Consumer";
          case S:
            var X = A.render;
            return A = A.displayName, A || (A = X.displayName || X.name || "", A = A !== "" ? "ForwardRef(" + A + ")" : "ForwardRef"), A;
          case R:
            return X = A.displayName || null, X !== null ? X : e(A.type) || "Memo";
          case k:
            X = A._payload, A = A._init;
            try {
              return e(A(X));
            } catch {
            }
        }
      return null;
    }
    function t(A) {
      return "" + A;
    }
    function r(A) {
      try {
        t(A);
        var X = !1;
      } catch {
        X = !0;
      }
      if (X) {
        X = console;
        var J = X.error, le = typeof Symbol == "function" && Symbol.toStringTag && A[Symbol.toStringTag] || A.constructor.name || "Object";
        return J.call(
          X,
          "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.",
          le
        ), t(A);
      }
    }
    function n() {
    }
    function i() {
      if (V === 0) {
        te = console.log, re = console.info, ae = console.warn, ne = console.error, F = console.group, H = console.groupCollapsed, ee = console.groupEnd;
        var A = {
          configurable: !0,
          enumerable: !0,
          value: n,
          writable: !0
        };
        Object.defineProperties(console, {
          info: A,
          log: A,
          warn: A,
          error: A,
          group: A,
          groupCollapsed: A,
          groupEnd: A
        });
      }
      V++;
    }
    function a() {
      if (V--, V === 0) {
        var A = { configurable: !0, enumerable: !0, writable: !0 };
        Object.defineProperties(console, {
          log: D({}, A, { value: te }),
          info: D({}, A, { value: re }),
          warn: D({}, A, { value: ae }),
          error: D({}, A, { value: ne }),
          group: D({}, A, { value: F }),
          groupCollapsed: D({}, A, { value: H }),
          groupEnd: D({}, A, { value: ee })
        });
      }
      0 > V && console.error(
        "disabledDepth fell below zero. This is a bug in React. Please file an issue."
      );
    }
    function o(A) {
      if (C === void 0)
        try {
          throw Error();
        } catch (J) {
          var X = J.stack.trim().match(/\n( *(at )?)/);
          C = X && X[1] || "", se = -1 < J.stack.indexOf(`
    at`) ? " (<anonymous>)" : -1 < J.stack.indexOf("@") ? "@unknown:0:0" : "";
        }
      return `
` + C + A + se;
    }
    function u(A, X) {
      if (!A || W) return "";
      var J = he.get(A);
      if (J !== void 0) return J;
      W = !0, J = Error.prepareStackTrace, Error.prepareStackTrace = void 0;
      var le = null;
      le = z.H, z.H = null, i();
      try {
        var Me = {
          DetermineComponentFrameRoot: function() {
            try {
              if (X) {
                var zt = function() {
                  throw Error();
                };
                if (Object.defineProperty(zt.prototype, "props", {
                  set: function() {
                    throw Error();
                  }
                }), typeof Reflect == "object" && Reflect.construct) {
                  try {
                    Reflect.construct(zt, []);
                  } catch (Et) {
                    var Ai = Et;
                  }
                  Reflect.construct(A, [], zt);
                } else {
                  try {
                    zt.call();
                  } catch (Et) {
                    Ai = Et;
                  }
                  A.call(zt.prototype);
                }
              } else {
                try {
                  throw Error();
                } catch (Et) {
                  Ai = Et;
                }
                (zt = A()) && typeof zt.catch == "function" && zt.catch(function() {
                });
              }
            } catch (Et) {
              if (Et && Ai && typeof Et.stack == "string")
                return [Et.stack, Ai.stack];
            }
            return [null, null];
          }
        };
        Me.DetermineComponentFrameRoot.displayName = "DetermineComponentFrameRoot";
        var _e = Object.getOwnPropertyDescriptor(
          Me.DetermineComponentFrameRoot,
          "name"
        );
        _e && _e.configurable && Object.defineProperty(
          Me.DetermineComponentFrameRoot,
          "name",
          { value: "DetermineComponentFrameRoot" }
        );
        var oe = Me.DetermineComponentFrameRoot(), At = oe[0], Or = oe[1];
        if (At && Or) {
          var Ge = At.split(`
`), nr = Or.split(`
`);
          for (oe = _e = 0; _e < Ge.length && !Ge[_e].includes(
            "DetermineComponentFrameRoot"
          ); )
            _e++;
          for (; oe < nr.length && !nr[oe].includes(
            "DetermineComponentFrameRoot"
          ); )
            oe++;
          if (_e === Ge.length || oe === nr.length)
            for (_e = Ge.length - 1, oe = nr.length - 1; 1 <= _e && 0 <= oe && Ge[_e] !== nr[oe]; )
              oe--;
          for (; 1 <= _e && 0 <= oe; _e--, oe--)
            if (Ge[_e] !== nr[oe]) {
              if (_e !== 1 || oe !== 1)
                do
                  if (_e--, oe--, 0 > oe || Ge[_e] !== nr[oe]) {
                    var yn = `
` + Ge[_e].replace(
                      " at new ",
                      " at "
                    );
                    return A.displayName && yn.includes("<anonymous>") && (yn = yn.replace("<anonymous>", A.displayName)), typeof A == "function" && he.set(A, yn), yn;
                  }
                while (1 <= _e && 0 <= oe);
              break;
            }
        }
      } finally {
        W = !1, z.H = le, a(), Error.prepareStackTrace = J;
      }
      return Ge = (Ge = A ? A.displayName || A.name : "") ? o(Ge) : "", typeof A == "function" && he.set(A, Ge), Ge;
    }
    function s(A) {
      if (A == null) return "";
      if (typeof A == "function") {
        var X = A.prototype;
        return u(
          A,
          !(!X || !X.isReactComponent)
        );
      }
      if (typeof A == "string") return o(A);
      switch (A) {
        case j:
          return o("Suspense");
        case M:
          return o("SuspenseList");
      }
      if (typeof A == "object")
        switch (A.$$typeof) {
          case S:
            return A = u(A.render, !1), A;
          case R:
            return s(A.type);
          case k:
            X = A._payload, A = A._init;
            try {
              return s(A(X));
            } catch {
            }
        }
      return "";
    }
    function c() {
      var A = z.A;
      return A === null ? null : A.getOwner();
    }
    function f(A) {
      if ($.call(A, "key")) {
        var X = Object.getOwnPropertyDescriptor(A, "key").get;
        if (X && X.isReactWarning) return !1;
      }
      return A.key !== void 0;
    }
    function l(A, X) {
      function J() {
        Ce || (Ce = !0, console.error(
          "%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)",
          X
        ));
      }
      J.isReactWarning = !0, Object.defineProperty(A, "key", {
        get: J,
        configurable: !0
      });
    }
    function d() {
      var A = e(this.type);
      return ct[A] || (ct[A] = !0, console.error(
        "Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."
      )), A = this.props.ref, A !== void 0 ? A : null;
    }
    function p(A, X, J, le, Me, _e) {
      return J = _e.ref, A = {
        $$typeof: O,
        type: A,
        key: X,
        props: _e,
        _owner: Me
      }, (J !== void 0 ? J : null) !== null ? Object.defineProperty(A, "ref", {
        enumerable: !1,
        get: d
      }) : Object.defineProperty(A, "ref", { enumerable: !1, value: null }), A._store = {}, Object.defineProperty(A._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: 0
      }), Object.defineProperty(A, "_debugInfo", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: null
      }), Object.freeze && (Object.freeze(A.props), Object.freeze(A)), A;
    }
    function y(A, X, J, le, Me, _e) {
      if (typeof A == "string" || typeof A == "function" || A === x || A === P || A === _ || A === j || A === M || A === q || typeof A == "object" && A !== null && (A.$$typeof === k || A.$$typeof === R || A.$$typeof === I || A.$$typeof === E || A.$$typeof === S || A.$$typeof === B || A.getModuleId !== void 0)) {
        var oe = X.children;
        if (oe !== void 0)
          if (le)
            if (G(oe)) {
              for (le = 0; le < oe.length; le++)
                v(oe[le], A);
              Object.freeze && Object.freeze(oe);
            } else
              console.error(
                "React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead."
              );
          else v(oe, A);
      } else
        oe = "", (A === void 0 || typeof A == "object" && A !== null && Object.keys(A).length === 0) && (oe += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports."), A === null ? le = "null" : G(A) ? le = "array" : A !== void 0 && A.$$typeof === O ? (le = "<" + (e(A.type) || "Unknown") + " />", oe = " Did you accidentally export a JSX literal instead of a component?") : le = typeof A, console.error(
          "React.jsx: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s",
          le,
          oe
        );
      if ($.call(X, "key")) {
        oe = e(A);
        var At = Object.keys(X).filter(function(Ge) {
          return Ge !== "key";
        });
        le = 0 < At.length ? "{key: someKey, " + At.join(": ..., ") + ": ...}" : "{key: someKey}", Pt[oe + le] || (At = 0 < At.length ? "{" + At.join(": ..., ") + ": ...}" : "{}", console.error(
          `A props object containing a "key" prop is being spread into JSX:
  let props = %s;
  <%s {...props} />
React keys must be passed directly to JSX without using spread:
  let props = %s;
  <%s key={someKey} {...props} />`,
          le,
          oe,
          At,
          oe
        ), Pt[oe + le] = !0);
      }
      if (oe = null, J !== void 0 && (r(J), oe = "" + J), f(X) && (r(X.key), oe = "" + X.key), "key" in X) {
        J = {};
        for (var Or in X)
          Or !== "key" && (J[Or] = X[Or]);
      } else J = X;
      return oe && l(
        J,
        typeof A == "function" ? A.displayName || A.name || "Unknown" : A
      ), p(A, oe, _e, Me, c(), J);
    }
    function v(A, X) {
      if (typeof A == "object" && A && A.$$typeof !== Oe) {
        if (G(A))
          for (var J = 0; J < A.length; J++) {
            var le = A[J];
            h(le) && g(le, X);
          }
        else if (h(A))
          A._store && (A._store.validated = 1);
        else if (A === null || typeof A != "object" ? J = null : (J = L && A[L] || A["@@iterator"], J = typeof J == "function" ? J : null), typeof J == "function" && J !== A.entries && (J = J.call(A), J !== A))
          for (; !(A = J.next()).done; )
            h(A.value) && g(A.value, X);
      }
    }
    function h(A) {
      return typeof A == "object" && A !== null && A.$$typeof === O;
    }
    function g(A, X) {
      if (A._store && !A._store.validated && A.key == null && (A._store.validated = 1, X = w(X), !rr[X])) {
        rr[X] = !0;
        var J = "";
        A && A._owner != null && A._owner !== c() && (J = null, typeof A._owner.tag == "number" ? J = e(A._owner.type) : typeof A._owner.name == "string" && (J = A._owner.name), J = " It was passed a child from " + J + ".");
        var le = z.getCurrentStack;
        z.getCurrentStack = function() {
          var Me = s(A.type);
          return le && (Me += le() || ""), Me;
        }, console.error(
          'Each child in a list should have a unique "key" prop.%s%s See https://react.dev/link/warning-keys for more information.',
          X,
          J
        ), z.getCurrentStack = le;
      }
    }
    function w(A) {
      var X = "", J = c();
      return J && (J = e(J.type)) && (X = `

Check the render method of \`` + J + "`."), X || (A = e(A)) && (X = `

Check the top-level render call using <` + A + ">."), X;
    }
    var b = T, O = Symbol.for("react.transitional.element"), m = Symbol.for("react.portal"), x = Symbol.for("react.fragment"), _ = Symbol.for("react.strict_mode"), P = Symbol.for("react.profiler"), E = Symbol.for("react.consumer"), I = Symbol.for("react.context"), S = Symbol.for("react.forward_ref"), j = Symbol.for("react.suspense"), M = Symbol.for("react.suspense_list"), R = Symbol.for("react.memo"), k = Symbol.for("react.lazy"), q = Symbol.for("react.offscreen"), L = Symbol.iterator, U = Symbol.for("react.client.reference"), z = b.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, $ = Object.prototype.hasOwnProperty, D = Object.assign, B = Symbol.for("react.client.reference"), G = Array.isArray, V = 0, te, re, ae, ne, F, H, ee;
    n.__reactDisabledLog = !0;
    var C, se, W = !1, he = new (typeof WeakMap == "function" ? WeakMap : Map)(), Oe = Symbol.for("react.client.reference"), Ce, ct = {}, Pt = {}, rr = {};
    gn.Fragment = x, gn.jsx = function(A, X, J, le, Me) {
      return y(A, X, J, !1, le, Me);
    }, gn.jsxs = function(A, X, J, le, Me) {
      return y(A, X, J, !0, le, Me);
    };
  }()), gn;
}
var up;
function GO() {
  return up || (up = 1, process.env.NODE_ENV === "production" ? Ti.exports = UO() : Ti.exports = WO()), Ti.exports;
}
var Y = GO();
function s0(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = s0(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function pe() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = s0(e)) && (n && (n += " "), n += t);
  return n;
}
var So, sp;
function Xe() {
  if (sp) return So;
  sp = 1;
  var e = Array.isArray;
  return So = e, So;
}
var Po, cp;
function c0() {
  if (cp) return Po;
  cp = 1;
  var e = typeof Ei == "object" && Ei && Ei.Object === Object && Ei;
  return Po = e, Po;
}
var Ao, lp;
function St() {
  if (lp) return Ao;
  lp = 1;
  var e = c0(), t = typeof self == "object" && self && self.Object === Object && self, r = e || t || Function("return this")();
  return Ao = r, Ao;
}
var Eo, fp;
function vi() {
  if (fp) return Eo;
  fp = 1;
  var e = St(), t = e.Symbol;
  return Eo = t, Eo;
}
var To, dp;
function HO() {
  if (dp) return To;
  dp = 1;
  var e = vi(), t = Object.prototype, r = t.hasOwnProperty, n = t.toString, i = e ? e.toStringTag : void 0;
  function a(o) {
    var u = r.call(o, i), s = o[i];
    try {
      o[i] = void 0;
      var c = !0;
    } catch {
    }
    var f = n.call(o);
    return c && (u ? o[i] = s : delete o[i]), f;
  }
  return To = a, To;
}
var jo, pp;
function KO() {
  if (pp) return jo;
  pp = 1;
  var e = Object.prototype, t = e.toString;
  function r(n) {
    return t.call(n);
  }
  return jo = r, jo;
}
var Co, hp;
function Lt() {
  if (hp) return Co;
  hp = 1;
  var e = vi(), t = HO(), r = KO(), n = "[object Null]", i = "[object Undefined]", a = e ? e.toStringTag : void 0;
  function o(u) {
    return u == null ? u === void 0 ? i : n : a && a in Object(u) ? t(u) : r(u);
  }
  return Co = o, Co;
}
var Mo, vp;
function Bt() {
  if (vp) return Mo;
  vp = 1;
  function e(t) {
    return t != null && typeof t == "object";
  }
  return Mo = e, Mo;
}
var Io, yp;
function un() {
  if (yp) return Io;
  yp = 1;
  var e = Lt(), t = Bt(), r = "[object Symbol]";
  function n(i) {
    return typeof i == "symbol" || t(i) && e(i) == r;
  }
  return Io = n, Io;
}
var $o, mp;
function Zf() {
  if (mp) return $o;
  mp = 1;
  var e = Xe(), t = un(), r = /\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/, n = /^\w*$/;
  function i(a, o) {
    if (e(a))
      return !1;
    var u = typeof a;
    return u == "number" || u == "symbol" || u == "boolean" || a == null || t(a) ? !0 : n.test(a) || !r.test(a) || o != null && a in Object(o);
  }
  return $o = i, $o;
}
var Ro, gp;
function Zt() {
  if (gp) return Ro;
  gp = 1;
  function e(t) {
    var r = typeof t;
    return t != null && (r == "object" || r == "function");
  }
  return Ro = e, Ro;
}
var ko, bp;
function Jf() {
  if (bp) return ko;
  bp = 1;
  var e = Lt(), t = Zt(), r = "[object AsyncFunction]", n = "[object Function]", i = "[object GeneratorFunction]", a = "[object Proxy]";
  function o(u) {
    if (!t(u))
      return !1;
    var s = e(u);
    return s == n || s == i || s == r || s == a;
  }
  return ko = o, ko;
}
var No, xp;
function VO() {
  if (xp) return No;
  xp = 1;
  var e = St(), t = e["__core-js_shared__"];
  return No = t, No;
}
var Do, wp;
function YO() {
  if (wp) return Do;
  wp = 1;
  var e = VO(), t = function() {
    var n = /[^.]+$/.exec(e && e.keys && e.keys.IE_PROTO || "");
    return n ? "Symbol(src)_1." + n : "";
  }();
  function r(n) {
    return !!t && t in n;
  }
  return Do = r, Do;
}
var qo, Op;
function l0() {
  if (Op) return qo;
  Op = 1;
  var e = Function.prototype, t = e.toString;
  function r(n) {
    if (n != null) {
      try {
        return t.call(n);
      } catch {
      }
      try {
        return n + "";
      } catch {
      }
    }
    return "";
  }
  return qo = r, qo;
}
var Lo, _p;
function XO() {
  if (_p) return Lo;
  _p = 1;
  var e = Jf(), t = YO(), r = Zt(), n = l0(), i = /[\\^$.*+?()[\]{}|]/g, a = /^\[object .+?Constructor\]$/, o = Function.prototype, u = Object.prototype, s = o.toString, c = u.hasOwnProperty, f = RegExp(
    "^" + s.call(c).replace(i, "\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$"
  );
  function l(d) {
    if (!r(d) || t(d))
      return !1;
    var p = e(d) ? f : a;
    return p.test(n(d));
  }
  return Lo = l, Lo;
}
var Bo, Sp;
function ZO() {
  if (Sp) return Bo;
  Sp = 1;
  function e(t, r) {
    return t == null ? void 0 : t[r];
  }
  return Bo = e, Bo;
}
var Fo, Pp;
function gr() {
  if (Pp) return Fo;
  Pp = 1;
  var e = XO(), t = ZO();
  function r(n, i) {
    var a = t(n, i);
    return e(a) ? a : void 0;
  }
  return Fo = r, Fo;
}
var zo, Ap;
function Ua() {
  if (Ap) return zo;
  Ap = 1;
  var e = gr(), t = e(Object, "create");
  return zo = t, zo;
}
var Uo, Ep;
function JO() {
  if (Ep) return Uo;
  Ep = 1;
  var e = Ua();
  function t() {
    this.__data__ = e ? e(null) : {}, this.size = 0;
  }
  return Uo = t, Uo;
}
var Wo, Tp;
function QO() {
  if (Tp) return Wo;
  Tp = 1;
  function e(t) {
    var r = this.has(t) && delete this.__data__[t];
    return this.size -= r ? 1 : 0, r;
  }
  return Wo = e, Wo;
}
var Go, jp;
function e_() {
  if (jp) return Go;
  jp = 1;
  var e = Ua(), t = "__lodash_hash_undefined__", r = Object.prototype, n = r.hasOwnProperty;
  function i(a) {
    var o = this.__data__;
    if (e) {
      var u = o[a];
      return u === t ? void 0 : u;
    }
    return n.call(o, a) ? o[a] : void 0;
  }
  return Go = i, Go;
}
var Ho, Cp;
function t_() {
  if (Cp) return Ho;
  Cp = 1;
  var e = Ua(), t = Object.prototype, r = t.hasOwnProperty;
  function n(i) {
    var a = this.__data__;
    return e ? a[i] !== void 0 : r.call(a, i);
  }
  return Ho = n, Ho;
}
var Ko, Mp;
function r_() {
  if (Mp) return Ko;
  Mp = 1;
  var e = Ua(), t = "__lodash_hash_undefined__";
  function r(n, i) {
    var a = this.__data__;
    return this.size += this.has(n) ? 0 : 1, a[n] = e && i === void 0 ? t : i, this;
  }
  return Ko = r, Ko;
}
var Vo, Ip;
function n_() {
  if (Ip) return Vo;
  Ip = 1;
  var e = JO(), t = QO(), r = e_(), n = t_(), i = r_();
  function a(o) {
    var u = -1, s = o == null ? 0 : o.length;
    for (this.clear(); ++u < s; ) {
      var c = o[u];
      this.set(c[0], c[1]);
    }
  }
  return a.prototype.clear = e, a.prototype.delete = t, a.prototype.get = r, a.prototype.has = n, a.prototype.set = i, Vo = a, Vo;
}
var Yo, $p;
function i_() {
  if ($p) return Yo;
  $p = 1;
  function e() {
    this.__data__ = [], this.size = 0;
  }
  return Yo = e, Yo;
}
var Xo, Rp;
function Qf() {
  if (Rp) return Xo;
  Rp = 1;
  function e(t, r) {
    return t === r || t !== t && r !== r;
  }
  return Xo = e, Xo;
}
var Zo, kp;
function Wa() {
  if (kp) return Zo;
  kp = 1;
  var e = Qf();
  function t(r, n) {
    for (var i = r.length; i--; )
      if (e(r[i][0], n))
        return i;
    return -1;
  }
  return Zo = t, Zo;
}
var Jo, Np;
function a_() {
  if (Np) return Jo;
  Np = 1;
  var e = Wa(), t = Array.prototype, r = t.splice;
  function n(i) {
    var a = this.__data__, o = e(a, i);
    if (o < 0)
      return !1;
    var u = a.length - 1;
    return o == u ? a.pop() : r.call(a, o, 1), --this.size, !0;
  }
  return Jo = n, Jo;
}
var Qo, Dp;
function o_() {
  if (Dp) return Qo;
  Dp = 1;
  var e = Wa();
  function t(r) {
    var n = this.__data__, i = e(n, r);
    return i < 0 ? void 0 : n[i][1];
  }
  return Qo = t, Qo;
}
var eu, qp;
function u_() {
  if (qp) return eu;
  qp = 1;
  var e = Wa();
  function t(r) {
    return e(this.__data__, r) > -1;
  }
  return eu = t, eu;
}
var tu, Lp;
function s_() {
  if (Lp) return tu;
  Lp = 1;
  var e = Wa();
  function t(r, n) {
    var i = this.__data__, a = e(i, r);
    return a < 0 ? (++this.size, i.push([r, n])) : i[a][1] = n, this;
  }
  return tu = t, tu;
}
var ru, Bp;
function Ga() {
  if (Bp) return ru;
  Bp = 1;
  var e = i_(), t = a_(), r = o_(), n = u_(), i = s_();
  function a(o) {
    var u = -1, s = o == null ? 0 : o.length;
    for (this.clear(); ++u < s; ) {
      var c = o[u];
      this.set(c[0], c[1]);
    }
  }
  return a.prototype.clear = e, a.prototype.delete = t, a.prototype.get = r, a.prototype.has = n, a.prototype.set = i, ru = a, ru;
}
var nu, Fp;
function ed() {
  if (Fp) return nu;
  Fp = 1;
  var e = gr(), t = St(), r = e(t, "Map");
  return nu = r, nu;
}
var iu, zp;
function c_() {
  if (zp) return iu;
  zp = 1;
  var e = n_(), t = Ga(), r = ed();
  function n() {
    this.size = 0, this.__data__ = {
      hash: new e(),
      map: new (r || t)(),
      string: new e()
    };
  }
  return iu = n, iu;
}
var au, Up;
function l_() {
  if (Up) return au;
  Up = 1;
  function e(t) {
    var r = typeof t;
    return r == "string" || r == "number" || r == "symbol" || r == "boolean" ? t !== "__proto__" : t === null;
  }
  return au = e, au;
}
var ou, Wp;
function Ha() {
  if (Wp) return ou;
  Wp = 1;
  var e = l_();
  function t(r, n) {
    var i = r.__data__;
    return e(n) ? i[typeof n == "string" ? "string" : "hash"] : i.map;
  }
  return ou = t, ou;
}
var uu, Gp;
function f_() {
  if (Gp) return uu;
  Gp = 1;
  var e = Ha();
  function t(r) {
    var n = e(this, r).delete(r);
    return this.size -= n ? 1 : 0, n;
  }
  return uu = t, uu;
}
var su, Hp;
function d_() {
  if (Hp) return su;
  Hp = 1;
  var e = Ha();
  function t(r) {
    return e(this, r).get(r);
  }
  return su = t, su;
}
var cu, Kp;
function p_() {
  if (Kp) return cu;
  Kp = 1;
  var e = Ha();
  function t(r) {
    return e(this, r).has(r);
  }
  return cu = t, cu;
}
var lu, Vp;
function h_() {
  if (Vp) return lu;
  Vp = 1;
  var e = Ha();
  function t(r, n) {
    var i = e(this, r), a = i.size;
    return i.set(r, n), this.size += i.size == a ? 0 : 1, this;
  }
  return lu = t, lu;
}
var fu, Yp;
function td() {
  if (Yp) return fu;
  Yp = 1;
  var e = c_(), t = f_(), r = d_(), n = p_(), i = h_();
  function a(o) {
    var u = -1, s = o == null ? 0 : o.length;
    for (this.clear(); ++u < s; ) {
      var c = o[u];
      this.set(c[0], c[1]);
    }
  }
  return a.prototype.clear = e, a.prototype.delete = t, a.prototype.get = r, a.prototype.has = n, a.prototype.set = i, fu = a, fu;
}
var du, Xp;
function f0() {
  if (Xp) return du;
  Xp = 1;
  var e = td(), t = "Expected a function";
  function r(n, i) {
    if (typeof n != "function" || i != null && typeof i != "function")
      throw new TypeError(t);
    var a = function() {
      var o = arguments, u = i ? i.apply(this, o) : o[0], s = a.cache;
      if (s.has(u))
        return s.get(u);
      var c = n.apply(this, o);
      return a.cache = s.set(u, c) || s, c;
    };
    return a.cache = new (r.Cache || e)(), a;
  }
  return r.Cache = e, du = r, du;
}
var pu, Zp;
function v_() {
  if (Zp) return pu;
  Zp = 1;
  var e = f0(), t = 500;
  function r(n) {
    var i = e(n, function(o) {
      return a.size === t && a.clear(), o;
    }), a = i.cache;
    return i;
  }
  return pu = r, pu;
}
var hu, Jp;
function y_() {
  if (Jp) return hu;
  Jp = 1;
  var e = v_(), t = /[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g, r = /\\(\\)?/g, n = e(function(i) {
    var a = [];
    return i.charCodeAt(0) === 46 && a.push(""), i.replace(t, function(o, u, s, c) {
      a.push(s ? c.replace(r, "$1") : u || o);
    }), a;
  });
  return hu = n, hu;
}
var vu, Qp;
function rd() {
  if (Qp) return vu;
  Qp = 1;
  function e(t, r) {
    for (var n = -1, i = t == null ? 0 : t.length, a = Array(i); ++n < i; )
      a[n] = r(t[n], n, t);
    return a;
  }
  return vu = e, vu;
}
var yu, eh;
function m_() {
  if (eh) return yu;
  eh = 1;
  var e = vi(), t = rd(), r = Xe(), n = un(), i = e ? e.prototype : void 0, a = i ? i.toString : void 0;
  function o(u) {
    if (typeof u == "string")
      return u;
    if (r(u))
      return t(u, o) + "";
    if (n(u))
      return a ? a.call(u) : "";
    var s = u + "";
    return s == "0" && 1 / u == -1 / 0 ? "-0" : s;
  }
  return yu = o, yu;
}
var mu, th;
function d0() {
  if (th) return mu;
  th = 1;
  var e = m_();
  function t(r) {
    return r == null ? "" : e(r);
  }
  return mu = t, mu;
}
var gu, rh;
function p0() {
  if (rh) return gu;
  rh = 1;
  var e = Xe(), t = Zf(), r = y_(), n = d0();
  function i(a, o) {
    return e(a) ? a : t(a, o) ? [a] : r(n(a));
  }
  return gu = i, gu;
}
var bu, nh;
function Ka() {
  if (nh) return bu;
  nh = 1;
  var e = un();
  function t(r) {
    if (typeof r == "string" || e(r))
      return r;
    var n = r + "";
    return n == "0" && 1 / r == -1 / 0 ? "-0" : n;
  }
  return bu = t, bu;
}
var xu, ih;
function nd() {
  if (ih) return xu;
  ih = 1;
  var e = p0(), t = Ka();
  function r(n, i) {
    i = e(i, n);
    for (var a = 0, o = i.length; n != null && a < o; )
      n = n[t(i[a++])];
    return a && a == o ? n : void 0;
  }
  return xu = r, xu;
}
var wu, ah;
function h0() {
  if (ah) return wu;
  ah = 1;
  var e = nd();
  function t(r, n, i) {
    var a = r == null ? void 0 : e(r, n);
    return a === void 0 ? i : a;
  }
  return wu = t, wu;
}
var g_ = h0();
const at = /* @__PURE__ */ Pe(g_);
var Ou, oh;
function b_() {
  if (oh) return Ou;
  oh = 1;
  function e(t) {
    return t == null;
  }
  return Ou = e, Ou;
}
var x_ = b_();
const ce = /* @__PURE__ */ Pe(x_);
var _u, uh;
function w_() {
  if (uh) return _u;
  uh = 1;
  var e = Lt(), t = Xe(), r = Bt(), n = "[object String]";
  function i(a) {
    return typeof a == "string" || !t(a) && r(a) && e(a) == n;
  }
  return _u = i, _u;
}
var O_ = w_();
const yi = /* @__PURE__ */ Pe(O_);
var __ = Jf();
const ue = /* @__PURE__ */ Pe(__);
var S_ = Zt();
const sn = /* @__PURE__ */ Pe(S_);
var ji = { exports: {} }, ve = {};
/**
 * @license React
 * react-is.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var sh;
function P_() {
  if (sh) return ve;
  sh = 1;
  var e = Symbol.for("react.element"), t = Symbol.for("react.portal"), r = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), i = Symbol.for("react.profiler"), a = Symbol.for("react.provider"), o = Symbol.for("react.context"), u = Symbol.for("react.server_context"), s = Symbol.for("react.forward_ref"), c = Symbol.for("react.suspense"), f = Symbol.for("react.suspense_list"), l = Symbol.for("react.memo"), d = Symbol.for("react.lazy"), p = Symbol.for("react.offscreen"), y;
  y = Symbol.for("react.module.reference");
  function v(h) {
    if (typeof h == "object" && h !== null) {
      var g = h.$$typeof;
      switch (g) {
        case e:
          switch (h = h.type, h) {
            case r:
            case i:
            case n:
            case c:
            case f:
              return h;
            default:
              switch (h = h && h.$$typeof, h) {
                case u:
                case o:
                case s:
                case d:
                case l:
                case a:
                  return h;
                default:
                  return g;
              }
          }
        case t:
          return g;
      }
    }
  }
  return ve.ContextConsumer = o, ve.ContextProvider = a, ve.Element = e, ve.ForwardRef = s, ve.Fragment = r, ve.Lazy = d, ve.Memo = l, ve.Portal = t, ve.Profiler = i, ve.StrictMode = n, ve.Suspense = c, ve.SuspenseList = f, ve.isAsyncMode = function() {
    return !1;
  }, ve.isConcurrentMode = function() {
    return !1;
  }, ve.isContextConsumer = function(h) {
    return v(h) === o;
  }, ve.isContextProvider = function(h) {
    return v(h) === a;
  }, ve.isElement = function(h) {
    return typeof h == "object" && h !== null && h.$$typeof === e;
  }, ve.isForwardRef = function(h) {
    return v(h) === s;
  }, ve.isFragment = function(h) {
    return v(h) === r;
  }, ve.isLazy = function(h) {
    return v(h) === d;
  }, ve.isMemo = function(h) {
    return v(h) === l;
  }, ve.isPortal = function(h) {
    return v(h) === t;
  }, ve.isProfiler = function(h) {
    return v(h) === i;
  }, ve.isStrictMode = function(h) {
    return v(h) === n;
  }, ve.isSuspense = function(h) {
    return v(h) === c;
  }, ve.isSuspenseList = function(h) {
    return v(h) === f;
  }, ve.isValidElementType = function(h) {
    return typeof h == "string" || typeof h == "function" || h === r || h === i || h === n || h === c || h === f || h === p || typeof h == "object" && h !== null && (h.$$typeof === d || h.$$typeof === l || h.$$typeof === a || h.$$typeof === o || h.$$typeof === s || h.$$typeof === y || h.getModuleId !== void 0);
  }, ve.typeOf = v, ve;
}
var ye = {};
/**
 * @license React
 * react-is.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var ch;
function A_() {
  return ch || (ch = 1, process.env.NODE_ENV !== "production" && function() {
    var e = Symbol.for("react.element"), t = Symbol.for("react.portal"), r = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), i = Symbol.for("react.profiler"), a = Symbol.for("react.provider"), o = Symbol.for("react.context"), u = Symbol.for("react.server_context"), s = Symbol.for("react.forward_ref"), c = Symbol.for("react.suspense"), f = Symbol.for("react.suspense_list"), l = Symbol.for("react.memo"), d = Symbol.for("react.lazy"), p = Symbol.for("react.offscreen"), y = !1, v = !1, h = !1, g = !1, w = !1, b;
    b = Symbol.for("react.module.reference");
    function O(W) {
      return !!(typeof W == "string" || typeof W == "function" || W === r || W === i || w || W === n || W === c || W === f || g || W === p || y || v || h || typeof W == "object" && W !== null && (W.$$typeof === d || W.$$typeof === l || W.$$typeof === a || W.$$typeof === o || W.$$typeof === s || // This needs to include all possible module reference object
      // types supported by any Flight configuration anywhere since
      // we don't know which Flight build this will end up being used
      // with.
      W.$$typeof === b || W.getModuleId !== void 0));
    }
    function m(W) {
      if (typeof W == "object" && W !== null) {
        var he = W.$$typeof;
        switch (he) {
          case e:
            var Oe = W.type;
            switch (Oe) {
              case r:
              case i:
              case n:
              case c:
              case f:
                return Oe;
              default:
                var Ce = Oe && Oe.$$typeof;
                switch (Ce) {
                  case u:
                  case o:
                  case s:
                  case d:
                  case l:
                  case a:
                    return Ce;
                  default:
                    return he;
                }
            }
          case t:
            return he;
        }
      }
    }
    var x = o, _ = a, P = e, E = s, I = r, S = d, j = l, M = t, R = i, k = n, q = c, L = f, U = !1, z = !1;
    function $(W) {
      return U || (U = !0, console.warn("The ReactIs.isAsyncMode() alias has been deprecated, and will be removed in React 18+.")), !1;
    }
    function D(W) {
      return z || (z = !0, console.warn("The ReactIs.isConcurrentMode() alias has been deprecated, and will be removed in React 18+.")), !1;
    }
    function B(W) {
      return m(W) === o;
    }
    function G(W) {
      return m(W) === a;
    }
    function V(W) {
      return typeof W == "object" && W !== null && W.$$typeof === e;
    }
    function te(W) {
      return m(W) === s;
    }
    function re(W) {
      return m(W) === r;
    }
    function ae(W) {
      return m(W) === d;
    }
    function ne(W) {
      return m(W) === l;
    }
    function F(W) {
      return m(W) === t;
    }
    function H(W) {
      return m(W) === i;
    }
    function ee(W) {
      return m(W) === n;
    }
    function C(W) {
      return m(W) === c;
    }
    function se(W) {
      return m(W) === f;
    }
    ye.ContextConsumer = x, ye.ContextProvider = _, ye.Element = P, ye.ForwardRef = E, ye.Fragment = I, ye.Lazy = S, ye.Memo = j, ye.Portal = M, ye.Profiler = R, ye.StrictMode = k, ye.Suspense = q, ye.SuspenseList = L, ye.isAsyncMode = $, ye.isConcurrentMode = D, ye.isContextConsumer = B, ye.isContextProvider = G, ye.isElement = V, ye.isForwardRef = te, ye.isFragment = re, ye.isLazy = ae, ye.isMemo = ne, ye.isPortal = F, ye.isProfiler = H, ye.isStrictMode = ee, ye.isSuspense = C, ye.isSuspenseList = se, ye.isValidElementType = O, ye.typeOf = m;
  }()), ye;
}
var lh;
function E_() {
  return lh || (lh = 1, process.env.NODE_ENV === "production" ? ji.exports = P_() : ji.exports = A_()), ji.exports;
}
var T_ = E_(), Su, fh;
function v0() {
  if (fh) return Su;
  fh = 1;
  var e = Lt(), t = Bt(), r = "[object Number]";
  function n(i) {
    return typeof i == "number" || t(i) && e(i) == r;
  }
  return Su = n, Su;
}
var Pu, dh;
function j_() {
  if (dh) return Pu;
  dh = 1;
  var e = v0();
  function t(r) {
    return e(r) && r != +r;
  }
  return Pu = t, Pu;
}
var C_ = j_();
const mi = /* @__PURE__ */ Pe(C_);
var M_ = v0();
const I_ = /* @__PURE__ */ Pe(M_);
var yt = function(t) {
  return t === 0 ? 0 : t > 0 ? 1 : -1;
}, Al = function(t) {
  return yi(t) && t.indexOf("%") === t.length - 1;
}, K = function(t) {
  return I_(t) && !mi(t);
}, ke = function(t) {
  return K(t) || yi(t);
}, $_ = 0, gi = function(t) {
  var r = ++$_;
  return "".concat(t || "").concat(r);
}, hr = function(t, r) {
  var n = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : 0, i = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : !1;
  if (!K(t) && !yi(t))
    return n;
  var a;
  if (Al(t)) {
    var o = t.indexOf("%");
    a = r * parseFloat(t.slice(0, o)) / 100;
  } else
    a = +t;
  return mi(a) && (a = n), i && a > r && (a = r), a;
}, Wt = function(t) {
  if (!t)
    return null;
  var r = Object.keys(t);
  return r && r.length ? t[r[0]] : null;
}, R_ = function(t) {
  if (!Array.isArray(t))
    return !1;
  for (var r = t.length, n = {}, i = 0; i < r; i++)
    if (!n[t[i]])
      n[t[i]] = !0;
    else
      return !0;
  return !1;
}, ht = function(t, r) {
  return K(t) && K(r) ? function(n) {
    return t + n * (r - t);
  } : function() {
    return r;
  };
};
function Wi(e, t, r) {
  return !e || !e.length ? null : e.find(function(n) {
    return n && (typeof t == "function" ? t(n) : at(n, t)) === r;
  });
}
function Mr(e, t) {
  for (var r in e)
    if ({}.hasOwnProperty.call(e, r) && (!{}.hasOwnProperty.call(t, r) || e[r] !== t[r]))
      return !1;
  for (var n in t)
    if ({}.hasOwnProperty.call(t, n) && !{}.hasOwnProperty.call(e, n))
      return !1;
  return !0;
}
function El(e) {
  "@babel/helpers - typeof";
  return El = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, El(e);
}
var k_ = ["viewBox", "children"], N_ = [
  "aria-activedescendant",
  "aria-atomic",
  "aria-autocomplete",
  "aria-busy",
  "aria-checked",
  "aria-colcount",
  "aria-colindex",
  "aria-colspan",
  "aria-controls",
  "aria-current",
  "aria-describedby",
  "aria-details",
  "aria-disabled",
  "aria-errormessage",
  "aria-expanded",
  "aria-flowto",
  "aria-haspopup",
  "aria-hidden",
  "aria-invalid",
  "aria-keyshortcuts",
  "aria-label",
  "aria-labelledby",
  "aria-level",
  "aria-live",
  "aria-modal",
  "aria-multiline",
  "aria-multiselectable",
  "aria-orientation",
  "aria-owns",
  "aria-placeholder",
  "aria-posinset",
  "aria-pressed",
  "aria-readonly",
  "aria-relevant",
  "aria-required",
  "aria-roledescription",
  "aria-rowcount",
  "aria-rowindex",
  "aria-rowspan",
  "aria-selected",
  "aria-setsize",
  "aria-sort",
  "aria-valuemax",
  "aria-valuemin",
  "aria-valuenow",
  "aria-valuetext",
  "className",
  "color",
  "height",
  "id",
  "lang",
  "max",
  "media",
  "method",
  "min",
  "name",
  "style",
  /*
   * removed 'type' SVGElementPropKey because we do not currently use any SVG elements
   * that can use it and it conflicts with the recharts prop 'type'
   * https://github.com/recharts/recharts/pull/3327
   * https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/type
   */
  // 'type',
  "target",
  "width",
  "role",
  "tabIndex",
  "accentHeight",
  "accumulate",
  "additive",
  "alignmentBaseline",
  "allowReorder",
  "alphabetic",
  "amplitude",
  "arabicForm",
  "ascent",
  "attributeName",
  "attributeType",
  "autoReverse",
  "azimuth",
  "baseFrequency",
  "baselineShift",
  "baseProfile",
  "bbox",
  "begin",
  "bias",
  "by",
  "calcMode",
  "capHeight",
  "clip",
  "clipPath",
  "clipPathUnits",
  "clipRule",
  "colorInterpolation",
  "colorInterpolationFilters",
  "colorProfile",
  "colorRendering",
  "contentScriptType",
  "contentStyleType",
  "cursor",
  "cx",
  "cy",
  "d",
  "decelerate",
  "descent",
  "diffuseConstant",
  "direction",
  "display",
  "divisor",
  "dominantBaseline",
  "dur",
  "dx",
  "dy",
  "edgeMode",
  "elevation",
  "enableBackground",
  "end",
  "exponent",
  "externalResourcesRequired",
  "fill",
  "fillOpacity",
  "fillRule",
  "filter",
  "filterRes",
  "filterUnits",
  "floodColor",
  "floodOpacity",
  "focusable",
  "fontFamily",
  "fontSize",
  "fontSizeAdjust",
  "fontStretch",
  "fontStyle",
  "fontVariant",
  "fontWeight",
  "format",
  "from",
  "fx",
  "fy",
  "g1",
  "g2",
  "glyphName",
  "glyphOrientationHorizontal",
  "glyphOrientationVertical",
  "glyphRef",
  "gradientTransform",
  "gradientUnits",
  "hanging",
  "horizAdvX",
  "horizOriginX",
  "href",
  "ideographic",
  "imageRendering",
  "in2",
  "in",
  "intercept",
  "k1",
  "k2",
  "k3",
  "k4",
  "k",
  "kernelMatrix",
  "kernelUnitLength",
  "kerning",
  "keyPoints",
  "keySplines",
  "keyTimes",
  "lengthAdjust",
  "letterSpacing",
  "lightingColor",
  "limitingConeAngle",
  "local",
  "markerEnd",
  "markerHeight",
  "markerMid",
  "markerStart",
  "markerUnits",
  "markerWidth",
  "mask",
  "maskContentUnits",
  "maskUnits",
  "mathematical",
  "mode",
  "numOctaves",
  "offset",
  "opacity",
  "operator",
  "order",
  "orient",
  "orientation",
  "origin",
  "overflow",
  "overlinePosition",
  "overlineThickness",
  "paintOrder",
  "panose1",
  "pathLength",
  "patternContentUnits",
  "patternTransform",
  "patternUnits",
  "pointerEvents",
  "pointsAtX",
  "pointsAtY",
  "pointsAtZ",
  "preserveAlpha",
  "preserveAspectRatio",
  "primitiveUnits",
  "r",
  "radius",
  "refX",
  "refY",
  "renderingIntent",
  "repeatCount",
  "repeatDur",
  "requiredExtensions",
  "requiredFeatures",
  "restart",
  "result",
  "rotate",
  "rx",
  "ry",
  "seed",
  "shapeRendering",
  "slope",
  "spacing",
  "specularConstant",
  "specularExponent",
  "speed",
  "spreadMethod",
  "startOffset",
  "stdDeviation",
  "stemh",
  "stemv",
  "stitchTiles",
  "stopColor",
  "stopOpacity",
  "strikethroughPosition",
  "strikethroughThickness",
  "string",
  "stroke",
  "strokeDasharray",
  "strokeDashoffset",
  "strokeLinecap",
  "strokeLinejoin",
  "strokeMiterlimit",
  "strokeOpacity",
  "strokeWidth",
  "surfaceScale",
  "systemLanguage",
  "tableValues",
  "targetX",
  "targetY",
  "textAnchor",
  "textDecoration",
  "textLength",
  "textRendering",
  "to",
  "transform",
  "u1",
  "u2",
  "underlinePosition",
  "underlineThickness",
  "unicode",
  "unicodeBidi",
  "unicodeRange",
  "unitsPerEm",
  "vAlphabetic",
  "values",
  "vectorEffect",
  "version",
  "vertAdvY",
  "vertOriginX",
  "vertOriginY",
  "vHanging",
  "vIdeographic",
  "viewTarget",
  "visibility",
  "vMathematical",
  "widths",
  "wordSpacing",
  "writingMode",
  "x1",
  "x2",
  "x",
  "xChannelSelector",
  "xHeight",
  "xlinkActuate",
  "xlinkArcrole",
  "xlinkHref",
  "xlinkRole",
  "xlinkShow",
  "xlinkTitle",
  "xlinkType",
  "xmlBase",
  "xmlLang",
  "xmlns",
  "xmlnsXlink",
  "xmlSpace",
  "y1",
  "y2",
  "y",
  "yChannelSelector",
  "z",
  "zoomAndPan",
  "ref",
  "key",
  "angle"
], ph = ["points", "pathLength"], Au = {
  svg: k_,
  polygon: ph,
  polyline: ph
}, id = ["dangerouslySetInnerHTML", "onCopy", "onCopyCapture", "onCut", "onCutCapture", "onPaste", "onPasteCapture", "onCompositionEnd", "onCompositionEndCapture", "onCompositionStart", "onCompositionStartCapture", "onCompositionUpdate", "onCompositionUpdateCapture", "onFocus", "onFocusCapture", "onBlur", "onBlurCapture", "onChange", "onChangeCapture", "onBeforeInput", "onBeforeInputCapture", "onInput", "onInputCapture", "onReset", "onResetCapture", "onSubmit", "onSubmitCapture", "onInvalid", "onInvalidCapture", "onLoad", "onLoadCapture", "onError", "onErrorCapture", "onKeyDown", "onKeyDownCapture", "onKeyPress", "onKeyPressCapture", "onKeyUp", "onKeyUpCapture", "onAbort", "onAbortCapture", "onCanPlay", "onCanPlayCapture", "onCanPlayThrough", "onCanPlayThroughCapture", "onDurationChange", "onDurationChangeCapture", "onEmptied", "onEmptiedCapture", "onEncrypted", "onEncryptedCapture", "onEnded", "onEndedCapture", "onLoadedData", "onLoadedDataCapture", "onLoadedMetadata", "onLoadedMetadataCapture", "onLoadStart", "onLoadStartCapture", "onPause", "onPauseCapture", "onPlay", "onPlayCapture", "onPlaying", "onPlayingCapture", "onProgress", "onProgressCapture", "onRateChange", "onRateChangeCapture", "onSeeked", "onSeekedCapture", "onSeeking", "onSeekingCapture", "onStalled", "onStalledCapture", "onSuspend", "onSuspendCapture", "onTimeUpdate", "onTimeUpdateCapture", "onVolumeChange", "onVolumeChangeCapture", "onWaiting", "onWaitingCapture", "onAuxClick", "onAuxClickCapture", "onClick", "onClickCapture", "onContextMenu", "onContextMenuCapture", "onDoubleClick", "onDoubleClickCapture", "onDrag", "onDragCapture", "onDragEnd", "onDragEndCapture", "onDragEnter", "onDragEnterCapture", "onDragExit", "onDragExitCapture", "onDragLeave", "onDragLeaveCapture", "onDragOver", "onDragOverCapture", "onDragStart", "onDragStartCapture", "onDrop", "onDropCapture", "onMouseDown", "onMouseDownCapture", "onMouseEnter", "onMouseLeave", "onMouseMove", "onMouseMoveCapture", "onMouseOut", "onMouseOutCapture", "onMouseOver", "onMouseOverCapture", "onMouseUp", "onMouseUpCapture", "onSelect", "onSelectCapture", "onTouchCancel", "onTouchCancelCapture", "onTouchEnd", "onTouchEndCapture", "onTouchMove", "onTouchMoveCapture", "onTouchStart", "onTouchStartCapture", "onPointerDown", "onPointerDownCapture", "onPointerMove", "onPointerMoveCapture", "onPointerUp", "onPointerUpCapture", "onPointerCancel", "onPointerCancelCapture", "onPointerEnter", "onPointerEnterCapture", "onPointerLeave", "onPointerLeaveCapture", "onPointerOver", "onPointerOverCapture", "onPointerOut", "onPointerOutCapture", "onGotPointerCapture", "onGotPointerCaptureCapture", "onLostPointerCapture", "onLostPointerCaptureCapture", "onScroll", "onScrollCapture", "onWheel", "onWheelCapture", "onAnimationStart", "onAnimationStartCapture", "onAnimationEnd", "onAnimationEndCapture", "onAnimationIteration", "onAnimationIterationCapture", "onTransitionEnd", "onTransitionEndCapture"], Gi = function(t, r) {
  if (!t || typeof t == "function" || typeof t == "boolean")
    return null;
  var n = t;
  if (/* @__PURE__ */ xt(t) && (n = t.props), !sn(n))
    return null;
  var i = {};
  return Object.keys(n).forEach(function(a) {
    id.includes(a) && (i[a] = r || function(o) {
      return n[a](n, o);
    });
  }), i;
}, D_ = function(t, r, n) {
  return function(i) {
    return t(r, n, i), null;
  };
}, Hi = function(t, r, n) {
  if (!sn(t) || El(t) !== "object")
    return null;
  var i = null;
  return Object.keys(t).forEach(function(a) {
    var o = t[a];
    id.includes(a) && typeof o == "function" && (i || (i = {}), i[a] = D_(o, r, n));
  }), i;
}, q_ = ["children"], L_ = ["children"];
function hh(e, t) {
  if (e == null) return {};
  var r = B_(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function B_(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function Tl(e) {
  "@babel/helpers - typeof";
  return Tl = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Tl(e);
}
var vh = {
  click: "onClick",
  mousedown: "onMouseDown",
  mouseup: "onMouseUp",
  mouseover: "onMouseOver",
  mousemove: "onMouseMove",
  mouseout: "onMouseOut",
  mouseenter: "onMouseEnter",
  mouseleave: "onMouseLeave",
  touchcancel: "onTouchCancel",
  touchend: "onTouchEnd",
  touchmove: "onTouchMove",
  touchstart: "onTouchStart",
  contextmenu: "onContextMenu",
  dblclick: "onDoubleClick"
}, Ht = function(t) {
  return typeof t == "string" ? t : t ? t.displayName || t.name || "Component" : "";
}, yh = null, Eu = null, ad = function e(t) {
  if (t === yh && Array.isArray(Eu))
    return Eu;
  var r = [];
  return fr.forEach(t, function(n) {
    ce(n) || (T_.isFragment(n) ? r = r.concat(e(n.props.children)) : r.push(n));
  }), Eu = r, yh = t, r;
};
function ot(e, t) {
  var r = [], n = [];
  return Array.isArray(t) ? n = t.map(function(i) {
    return Ht(i);
  }) : n = [Ht(t)], ad(e).forEach(function(i) {
    var a = at(i, "type.displayName") || at(i, "type.name");
    n.indexOf(a) !== -1 && r.push(i);
  }), r;
}
function Qe(e, t) {
  var r = ot(e, t);
  return r[0];
}
var mh = function(t) {
  if (!t || !t.props)
    return !1;
  var r = t.props, n = r.width, i = r.height;
  return !(!K(n) || n <= 0 || !K(i) || i <= 0);
}, F_ = ["a", "altGlyph", "altGlyphDef", "altGlyphItem", "animate", "animateColor", "animateMotion", "animateTransform", "circle", "clipPath", "color-profile", "cursor", "defs", "desc", "ellipse", "feBlend", "feColormatrix", "feComponentTransfer", "feComposite", "feConvolveMatrix", "feDiffuseLighting", "feDisplacementMap", "feDistantLight", "feFlood", "feFuncA", "feFuncB", "feFuncG", "feFuncR", "feGaussianBlur", "feImage", "feMerge", "feMergeNode", "feMorphology", "feOffset", "fePointLight", "feSpecularLighting", "feSpotLight", "feTile", "feTurbulence", "filter", "font", "font-face", "font-face-format", "font-face-name", "font-face-url", "foreignObject", "g", "glyph", "glyphRef", "hkern", "image", "line", "lineGradient", "marker", "mask", "metadata", "missing-glyph", "mpath", "path", "pattern", "polygon", "polyline", "radialGradient", "rect", "script", "set", "stop", "style", "svg", "switch", "symbol", "text", "textPath", "title", "tref", "tspan", "use", "view", "vkern"], z_ = function(t) {
  return t && t.type && yi(t.type) && F_.indexOf(t.type) >= 0;
}, U_ = function(t) {
  return t && Tl(t) === "object" && "clipDot" in t;
}, W_ = function(t, r, n, i) {
  var a, o = (a = Au == null ? void 0 : Au[i]) !== null && a !== void 0 ? a : [];
  return !ue(t) && (i && o.includes(r) || N_.includes(r)) || n && id.includes(r);
}, fe = function(t, r, n) {
  if (!t || typeof t == "function" || typeof t == "boolean")
    return null;
  var i = t;
  if (/* @__PURE__ */ xt(t) && (i = t.props), !sn(i))
    return null;
  var a = {};
  return Object.keys(i).forEach(function(o) {
    var u;
    W_((u = i) === null || u === void 0 ? void 0 : u[o], o, r, n) && (a[o] = i[o]);
  }), a;
}, jl = function e(t, r) {
  if (t === r)
    return !0;
  var n = fr.count(t);
  if (n !== fr.count(r))
    return !1;
  if (n === 0)
    return !0;
  if (n === 1)
    return gh(Array.isArray(t) ? t[0] : t, Array.isArray(r) ? r[0] : r);
  for (var i = 0; i < n; i++) {
    var a = t[i], o = r[i];
    if (Array.isArray(a) || Array.isArray(o)) {
      if (!e(a, o))
        return !1;
    } else if (!gh(a, o))
      return !1;
  }
  return !0;
}, gh = function(t, r) {
  if (ce(t) && ce(r))
    return !0;
  if (!ce(t) && !ce(r)) {
    var n = t.props || {}, i = n.children, a = hh(n, q_), o = r.props || {}, u = o.children, s = hh(o, L_);
    return i && u ? Mr(a, s) && jl(i, u) : !i && !u ? Mr(a, s) : !1;
  }
  return !1;
}, bh = function(t, r) {
  var n = [], i = {};
  return ad(t).forEach(function(a, o) {
    if (z_(a))
      n.push(a);
    else if (a) {
      var u = Ht(a.type), s = r[u] || {}, c = s.handler, f = s.once;
      if (c && (!f || !i[u])) {
        var l = c(a, u, o);
        n.push(l), i[u] = !0;
      }
    }
  }), n;
}, G_ = function(t) {
  var r = t && t.type;
  return r && vh[r] ? vh[r] : null;
}, H_ = function(t, r) {
  return ad(r).indexOf(t);
}, K_ = ["children", "width", "height", "viewBox", "className", "style", "title", "desc"];
function Cl() {
  return Cl = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Cl.apply(this, arguments);
}
function V_(e, t) {
  if (e == null) return {};
  var r = Y_(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function Y_(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function Ml(e) {
  var t = e.children, r = e.width, n = e.height, i = e.viewBox, a = e.className, o = e.style, u = e.title, s = e.desc, c = V_(e, K_), f = i || {
    width: r,
    height: n,
    x: 0,
    y: 0
  }, l = pe("recharts-surface", a);
  return /* @__PURE__ */ T.createElement("svg", Cl({}, fe(c, !0, "svg"), {
    className: l,
    width: r,
    height: n,
    style: o,
    viewBox: "".concat(f.x, " ").concat(f.y, " ").concat(f.width, " ").concat(f.height)
  }), /* @__PURE__ */ T.createElement("title", null, u), /* @__PURE__ */ T.createElement("desc", null, s), t);
}
var X_ = ["children", "className"];
function Il() {
  return Il = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Il.apply(this, arguments);
}
function Z_(e, t) {
  if (e == null) return {};
  var r = J_(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function J_(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
var je = /* @__PURE__ */ T.forwardRef(function(e, t) {
  var r = e.children, n = e.className, i = Z_(e, X_), a = pe("recharts-layer", n);
  return /* @__PURE__ */ T.createElement("g", Il({
    className: a
  }, fe(i, !0), {
    ref: t
  }), r);
}), Q_ = process.env.NODE_ENV !== "production", kr = function(t, r) {
  for (var n = arguments.length, i = new Array(n > 2 ? n - 2 : 0), a = 2; a < n; a++)
    i[a - 2] = arguments[a];
  if (Q_ && typeof console < "u" && console.warn && (r === void 0 && console.warn("LogUtils requires an error message argument"), !t))
    if (r === void 0)
      console.warn("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");
    else {
      var o = 0;
      console.warn(r.replace(/%s/g, function() {
        return i[o++];
      }));
    }
}, Tu, xh;
function e1() {
  if (xh) return Tu;
  xh = 1;
  function e(t, r, n) {
    var i = -1, a = t.length;
    r < 0 && (r = -r > a ? 0 : a + r), n = n > a ? a : n, n < 0 && (n += a), a = r > n ? 0 : n - r >>> 0, r >>>= 0;
    for (var o = Array(a); ++i < a; )
      o[i] = t[i + r];
    return o;
  }
  return Tu = e, Tu;
}
var ju, wh;
function t1() {
  if (wh) return ju;
  wh = 1;
  var e = e1();
  function t(r, n, i) {
    var a = r.length;
    return i = i === void 0 ? a : i, !n && i >= a ? r : e(r, n, i);
  }
  return ju = t, ju;
}
var Cu, Oh;
function y0() {
  if (Oh) return Cu;
  Oh = 1;
  var e = "\\ud800-\\udfff", t = "\\u0300-\\u036f", r = "\\ufe20-\\ufe2f", n = "\\u20d0-\\u20ff", i = t + r + n, a = "\\ufe0e\\ufe0f", o = "\\u200d", u = RegExp("[" + o + e + i + a + "]");
  function s(c) {
    return u.test(c);
  }
  return Cu = s, Cu;
}
var Mu, _h;
function r1() {
  if (_h) return Mu;
  _h = 1;
  function e(t) {
    return t.split("");
  }
  return Mu = e, Mu;
}
var Iu, Sh;
function n1() {
  if (Sh) return Iu;
  Sh = 1;
  var e = "\\ud800-\\udfff", t = "\\u0300-\\u036f", r = "\\ufe20-\\ufe2f", n = "\\u20d0-\\u20ff", i = t + r + n, a = "\\ufe0e\\ufe0f", o = "[" + e + "]", u = "[" + i + "]", s = "\\ud83c[\\udffb-\\udfff]", c = "(?:" + u + "|" + s + ")", f = "[^" + e + "]", l = "(?:\\ud83c[\\udde6-\\uddff]){2}", d = "[\\ud800-\\udbff][\\udc00-\\udfff]", p = "\\u200d", y = c + "?", v = "[" + a + "]?", h = "(?:" + p + "(?:" + [f, l, d].join("|") + ")" + v + y + ")*", g = v + y + h, w = "(?:" + [f + u + "?", u, l, d, o].join("|") + ")", b = RegExp(s + "(?=" + s + ")|" + w + g, "g");
  function O(m) {
    return m.match(b) || [];
  }
  return Iu = O, Iu;
}
var $u, Ph;
function i1() {
  if (Ph) return $u;
  Ph = 1;
  var e = r1(), t = y0(), r = n1();
  function n(i) {
    return t(i) ? r(i) : e(i);
  }
  return $u = n, $u;
}
var Ru, Ah;
function a1() {
  if (Ah) return Ru;
  Ah = 1;
  var e = t1(), t = y0(), r = i1(), n = d0();
  function i(a) {
    return function(o) {
      o = n(o);
      var u = t(o) ? r(o) : void 0, s = u ? u[0] : o.charAt(0), c = u ? e(u, 1).join("") : o.slice(1);
      return s[a]() + c;
    };
  }
  return Ru = i, Ru;
}
var ku, Eh;
function o1() {
  if (Eh) return ku;
  Eh = 1;
  var e = a1(), t = e("toUpperCase");
  return ku = t, ku;
}
var u1 = o1();
const Va = /* @__PURE__ */ Pe(u1);
function Se(e) {
  return function() {
    return e;
  };
}
const m0 = Math.cos, Ki = Math.sin, mt = Math.sqrt, Vi = Math.PI, Ya = 2 * Vi, $l = Math.PI, Rl = 2 * $l, ar = 1e-6, s1 = Rl - ar;
function g0(e) {
  this._ += e[0];
  for (let t = 1, r = e.length; t < r; ++t)
    this._ += arguments[t] + e[t];
}
function c1(e) {
  let t = Math.floor(e);
  if (!(t >= 0)) throw new Error(`invalid digits: ${e}`);
  if (t > 15) return g0;
  const r = 10 ** t;
  return function(n) {
    this._ += n[0];
    for (let i = 1, a = n.length; i < a; ++i)
      this._ += Math.round(arguments[i] * r) / r + n[i];
  };
}
class l1 {
  constructor(t) {
    this._x0 = this._y0 = // start of current subpath
    this._x1 = this._y1 = null, this._ = "", this._append = t == null ? g0 : c1(t);
  }
  moveTo(t, r) {
    this._append`M${this._x0 = this._x1 = +t},${this._y0 = this._y1 = +r}`;
  }
  closePath() {
    this._x1 !== null && (this._x1 = this._x0, this._y1 = this._y0, this._append`Z`);
  }
  lineTo(t, r) {
    this._append`L${this._x1 = +t},${this._y1 = +r}`;
  }
  quadraticCurveTo(t, r, n, i) {
    this._append`Q${+t},${+r},${this._x1 = +n},${this._y1 = +i}`;
  }
  bezierCurveTo(t, r, n, i, a, o) {
    this._append`C${+t},${+r},${+n},${+i},${this._x1 = +a},${this._y1 = +o}`;
  }
  arcTo(t, r, n, i, a) {
    if (t = +t, r = +r, n = +n, i = +i, a = +a, a < 0) throw new Error(`negative radius: ${a}`);
    let o = this._x1, u = this._y1, s = n - t, c = i - r, f = o - t, l = u - r, d = f * f + l * l;
    if (this._x1 === null)
      this._append`M${this._x1 = t},${this._y1 = r}`;
    else if (d > ar) if (!(Math.abs(l * s - c * f) > ar) || !a)
      this._append`L${this._x1 = t},${this._y1 = r}`;
    else {
      let p = n - o, y = i - u, v = s * s + c * c, h = p * p + y * y, g = Math.sqrt(v), w = Math.sqrt(d), b = a * Math.tan(($l - Math.acos((v + d - h) / (2 * g * w))) / 2), O = b / w, m = b / g;
      Math.abs(O - 1) > ar && this._append`L${t + O * f},${r + O * l}`, this._append`A${a},${a},0,0,${+(l * p > f * y)},${this._x1 = t + m * s},${this._y1 = r + m * c}`;
    }
  }
  arc(t, r, n, i, a, o) {
    if (t = +t, r = +r, n = +n, o = !!o, n < 0) throw new Error(`negative radius: ${n}`);
    let u = n * Math.cos(i), s = n * Math.sin(i), c = t + u, f = r + s, l = 1 ^ o, d = o ? i - a : a - i;
    this._x1 === null ? this._append`M${c},${f}` : (Math.abs(this._x1 - c) > ar || Math.abs(this._y1 - f) > ar) && this._append`L${c},${f}`, n && (d < 0 && (d = d % Rl + Rl), d > s1 ? this._append`A${n},${n},0,1,${l},${t - u},${r - s}A${n},${n},0,1,${l},${this._x1 = c},${this._y1 = f}` : d > ar && this._append`A${n},${n},0,${+(d >= $l)},${l},${this._x1 = t + n * Math.cos(a)},${this._y1 = r + n * Math.sin(a)}`);
  }
  rect(t, r, n, i) {
    this._append`M${this._x0 = this._x1 = +t},${this._y0 = this._y1 = +r}h${n = +n}v${+i}h${-n}Z`;
  }
  toString() {
    return this._;
  }
}
function od(e) {
  let t = 3;
  return e.digits = function(r) {
    if (!arguments.length) return t;
    if (r == null)
      t = null;
    else {
      const n = Math.floor(r);
      if (!(n >= 0)) throw new RangeError(`invalid digits: ${r}`);
      t = n;
    }
    return e;
  }, () => new l1(t);
}
function ud(e) {
  return typeof e == "object" && "length" in e ? e : Array.from(e);
}
function b0(e) {
  this._context = e;
}
b0.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
      // falls through
      default:
        this._context.lineTo(e, t);
        break;
    }
  }
};
function Xa(e) {
  return new b0(e);
}
function x0(e) {
  return e[0];
}
function w0(e) {
  return e[1];
}
function O0(e, t) {
  var r = Se(!0), n = null, i = Xa, a = null, o = od(u);
  e = typeof e == "function" ? e : e === void 0 ? x0 : Se(e), t = typeof t == "function" ? t : t === void 0 ? w0 : Se(t);
  function u(s) {
    var c, f = (s = ud(s)).length, l, d = !1, p;
    for (n == null && (a = i(p = o())), c = 0; c <= f; ++c)
      !(c < f && r(l = s[c], c, s)) === d && ((d = !d) ? a.lineStart() : a.lineEnd()), d && a.point(+e(l, c, s), +t(l, c, s));
    if (p) return a = null, p + "" || null;
  }
  return u.x = function(s) {
    return arguments.length ? (e = typeof s == "function" ? s : Se(+s), u) : e;
  }, u.y = function(s) {
    return arguments.length ? (t = typeof s == "function" ? s : Se(+s), u) : t;
  }, u.defined = function(s) {
    return arguments.length ? (r = typeof s == "function" ? s : Se(!!s), u) : r;
  }, u.curve = function(s) {
    return arguments.length ? (i = s, n != null && (a = i(n)), u) : i;
  }, u.context = function(s) {
    return arguments.length ? (s == null ? n = a = null : a = i(n = s), u) : n;
  }, u;
}
function Ci(e, t, r) {
  var n = null, i = Se(!0), a = null, o = Xa, u = null, s = od(c);
  e = typeof e == "function" ? e : e === void 0 ? x0 : Se(+e), t = typeof t == "function" ? t : Se(t === void 0 ? 0 : +t), r = typeof r == "function" ? r : r === void 0 ? w0 : Se(+r);
  function c(l) {
    var d, p, y, v = (l = ud(l)).length, h, g = !1, w, b = new Array(v), O = new Array(v);
    for (a == null && (u = o(w = s())), d = 0; d <= v; ++d) {
      if (!(d < v && i(h = l[d], d, l)) === g)
        if (g = !g)
          p = d, u.areaStart(), u.lineStart();
        else {
          for (u.lineEnd(), u.lineStart(), y = d - 1; y >= p; --y)
            u.point(b[y], O[y]);
          u.lineEnd(), u.areaEnd();
        }
      g && (b[d] = +e(h, d, l), O[d] = +t(h, d, l), u.point(n ? +n(h, d, l) : b[d], r ? +r(h, d, l) : O[d]));
    }
    if (w) return u = null, w + "" || null;
  }
  function f() {
    return O0().defined(i).curve(o).context(a);
  }
  return c.x = function(l) {
    return arguments.length ? (e = typeof l == "function" ? l : Se(+l), n = null, c) : e;
  }, c.x0 = function(l) {
    return arguments.length ? (e = typeof l == "function" ? l : Se(+l), c) : e;
  }, c.x1 = function(l) {
    return arguments.length ? (n = l == null ? null : typeof l == "function" ? l : Se(+l), c) : n;
  }, c.y = function(l) {
    return arguments.length ? (t = typeof l == "function" ? l : Se(+l), r = null, c) : t;
  }, c.y0 = function(l) {
    return arguments.length ? (t = typeof l == "function" ? l : Se(+l), c) : t;
  }, c.y1 = function(l) {
    return arguments.length ? (r = l == null ? null : typeof l == "function" ? l : Se(+l), c) : r;
  }, c.lineX0 = c.lineY0 = function() {
    return f().x(e).y(t);
  }, c.lineY1 = function() {
    return f().x(e).y(r);
  }, c.lineX1 = function() {
    return f().x(n).y(t);
  }, c.defined = function(l) {
    return arguments.length ? (i = typeof l == "function" ? l : Se(!!l), c) : i;
  }, c.curve = function(l) {
    return arguments.length ? (o = l, a != null && (u = o(a)), c) : o;
  }, c.context = function(l) {
    return arguments.length ? (l == null ? a = u = null : u = o(a = l), c) : a;
  }, c;
}
class _0 {
  constructor(t, r) {
    this._context = t, this._x = r;
  }
  areaStart() {
    this._line = 0;
  }
  areaEnd() {
    this._line = NaN;
  }
  lineStart() {
    this._point = 0;
  }
  lineEnd() {
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  }
  point(t, r) {
    switch (t = +t, r = +r, this._point) {
      case 0: {
        this._point = 1, this._line ? this._context.lineTo(t, r) : this._context.moveTo(t, r);
        break;
      }
      case 1:
        this._point = 2;
      // falls through
      default: {
        this._x ? this._context.bezierCurveTo(this._x0 = (this._x0 + t) / 2, this._y0, this._x0, r, t, r) : this._context.bezierCurveTo(this._x0, this._y0 = (this._y0 + r) / 2, t, this._y0, t, r);
        break;
      }
    }
    this._x0 = t, this._y0 = r;
  }
}
function f1(e) {
  return new _0(e, !0);
}
function d1(e) {
  return new _0(e, !1);
}
const sd = {
  draw(e, t) {
    const r = mt(t / Vi);
    e.moveTo(r, 0), e.arc(0, 0, r, 0, Ya);
  }
}, p1 = {
  draw(e, t) {
    const r = mt(t / 5) / 2;
    e.moveTo(-3 * r, -r), e.lineTo(-r, -r), e.lineTo(-r, -3 * r), e.lineTo(r, -3 * r), e.lineTo(r, -r), e.lineTo(3 * r, -r), e.lineTo(3 * r, r), e.lineTo(r, r), e.lineTo(r, 3 * r), e.lineTo(-r, 3 * r), e.lineTo(-r, r), e.lineTo(-3 * r, r), e.closePath();
  }
}, S0 = mt(1 / 3), h1 = S0 * 2, v1 = {
  draw(e, t) {
    const r = mt(t / h1), n = r * S0;
    e.moveTo(0, -r), e.lineTo(n, 0), e.lineTo(0, r), e.lineTo(-n, 0), e.closePath();
  }
}, y1 = {
  draw(e, t) {
    const r = mt(t), n = -r / 2;
    e.rect(n, n, r, r);
  }
}, m1 = 0.8908130915292852, P0 = Ki(Vi / 10) / Ki(7 * Vi / 10), g1 = Ki(Ya / 10) * P0, b1 = -m0(Ya / 10) * P0, x1 = {
  draw(e, t) {
    const r = mt(t * m1), n = g1 * r, i = b1 * r;
    e.moveTo(0, -r), e.lineTo(n, i);
    for (let a = 1; a < 5; ++a) {
      const o = Ya * a / 5, u = m0(o), s = Ki(o);
      e.lineTo(s * r, -u * r), e.lineTo(u * n - s * i, s * n + u * i);
    }
    e.closePath();
  }
}, Nu = mt(3), w1 = {
  draw(e, t) {
    const r = -mt(t / (Nu * 3));
    e.moveTo(0, r * 2), e.lineTo(-Nu * r, -r), e.lineTo(Nu * r, -r), e.closePath();
  }
}, rt = -0.5, nt = mt(3) / 2, kl = 1 / mt(12), O1 = (kl / 2 + 1) * 3, _1 = {
  draw(e, t) {
    const r = mt(t / O1), n = r / 2, i = r * kl, a = n, o = r * kl + r, u = -a, s = o;
    e.moveTo(n, i), e.lineTo(a, o), e.lineTo(u, s), e.lineTo(rt * n - nt * i, nt * n + rt * i), e.lineTo(rt * a - nt * o, nt * a + rt * o), e.lineTo(rt * u - nt * s, nt * u + rt * s), e.lineTo(rt * n + nt * i, rt * i - nt * n), e.lineTo(rt * a + nt * o, rt * o - nt * a), e.lineTo(rt * u + nt * s, rt * s - nt * u), e.closePath();
  }
};
function S1(e, t) {
  let r = null, n = od(i);
  e = typeof e == "function" ? e : Se(e || sd), t = typeof t == "function" ? t : Se(t === void 0 ? 64 : +t);
  function i() {
    let a;
    if (r || (r = a = n()), e.apply(this, arguments).draw(r, +t.apply(this, arguments)), a) return r = null, a + "" || null;
  }
  return i.type = function(a) {
    return arguments.length ? (e = typeof a == "function" ? a : Se(a), i) : e;
  }, i.size = function(a) {
    return arguments.length ? (t = typeof a == "function" ? a : Se(+a), i) : t;
  }, i.context = function(a) {
    return arguments.length ? (r = a ?? null, i) : r;
  }, i;
}
function Yi() {
}
function Xi(e, t, r) {
  e._context.bezierCurveTo(
    (2 * e._x0 + e._x1) / 3,
    (2 * e._y0 + e._y1) / 3,
    (e._x0 + 2 * e._x1) / 3,
    (e._y0 + 2 * e._y1) / 3,
    (e._x0 + 4 * e._x1 + t) / 6,
    (e._y0 + 4 * e._y1 + r) / 6
  );
}
function A0(e) {
  this._context = e;
}
A0.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 3:
        Xi(this, this._x1, this._y1);
      // falls through
      case 2:
        this._context.lineTo(this._x1, this._y1);
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3, this._context.lineTo((5 * this._x0 + this._x1) / 6, (5 * this._y0 + this._y1) / 6);
      // falls through
      default:
        Xi(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function P1(e) {
  return new A0(e);
}
function E0(e) {
  this._context = e;
}
E0.prototype = {
  areaStart: Yi,
  areaEnd: Yi,
  lineStart: function() {
    this._x0 = this._x1 = this._x2 = this._x3 = this._x4 = this._y0 = this._y1 = this._y2 = this._y3 = this._y4 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 1: {
        this._context.moveTo(this._x2, this._y2), this._context.closePath();
        break;
      }
      case 2: {
        this._context.moveTo((this._x2 + 2 * this._x3) / 3, (this._y2 + 2 * this._y3) / 3), this._context.lineTo((this._x3 + 2 * this._x2) / 3, (this._y3 + 2 * this._y2) / 3), this._context.closePath();
        break;
      }
      case 3: {
        this.point(this._x2, this._y2), this.point(this._x3, this._y3), this.point(this._x4, this._y4);
        break;
      }
    }
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._x2 = e, this._y2 = t;
        break;
      case 1:
        this._point = 2, this._x3 = e, this._y3 = t;
        break;
      case 2:
        this._point = 3, this._x4 = e, this._y4 = t, this._context.moveTo((this._x0 + 4 * this._x1 + e) / 6, (this._y0 + 4 * this._y1 + t) / 6);
        break;
      default:
        Xi(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function A1(e) {
  return new E0(e);
}
function T0(e) {
  this._context = e;
}
T0.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = NaN, this._point = 0;
  },
  lineEnd: function() {
    (this._line || this._line !== 0 && this._point === 3) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1;
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3;
        var r = (this._x0 + 4 * this._x1 + e) / 6, n = (this._y0 + 4 * this._y1 + t) / 6;
        this._line ? this._context.lineTo(r, n) : this._context.moveTo(r, n);
        break;
      case 3:
        this._point = 4;
      // falls through
      default:
        Xi(this, e, t);
        break;
    }
    this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t;
  }
};
function E1(e) {
  return new T0(e);
}
function j0(e) {
  this._context = e;
}
j0.prototype = {
  areaStart: Yi,
  areaEnd: Yi,
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    this._point && this._context.closePath();
  },
  point: function(e, t) {
    e = +e, t = +t, this._point ? this._context.lineTo(e, t) : (this._point = 1, this._context.moveTo(e, t));
  }
};
function T1(e) {
  return new j0(e);
}
function Th(e) {
  return e < 0 ? -1 : 1;
}
function jh(e, t, r) {
  var n = e._x1 - e._x0, i = t - e._x1, a = (e._y1 - e._y0) / (n || i < 0 && -0), o = (r - e._y1) / (i || n < 0 && -0), u = (a * i + o * n) / (n + i);
  return (Th(a) + Th(o)) * Math.min(Math.abs(a), Math.abs(o), 0.5 * Math.abs(u)) || 0;
}
function Ch(e, t) {
  var r = e._x1 - e._x0;
  return r ? (3 * (e._y1 - e._y0) / r - t) / 2 : t;
}
function Du(e, t, r) {
  var n = e._x0, i = e._y0, a = e._x1, o = e._y1, u = (a - n) / 3;
  e._context.bezierCurveTo(n + u, i + u * t, a - u, o - u * r, a, o);
}
function Zi(e) {
  this._context = e;
}
Zi.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = this._t0 = NaN, this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 2:
        this._context.lineTo(this._x1, this._y1);
        break;
      case 3:
        Du(this, this._t0, Ch(this, this._t0));
        break;
    }
    (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line = 1 - this._line;
  },
  point: function(e, t) {
    var r = NaN;
    if (e = +e, t = +t, !(e === this._x1 && t === this._y1)) {
      switch (this._point) {
        case 0:
          this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
          break;
        case 1:
          this._point = 2;
          break;
        case 2:
          this._point = 3, Du(this, Ch(this, r = jh(this, e, t)), r);
          break;
        default:
          Du(this, this._t0, r = jh(this, e, t));
          break;
      }
      this._x0 = this._x1, this._x1 = e, this._y0 = this._y1, this._y1 = t, this._t0 = r;
    }
  }
};
function C0(e) {
  this._context = new M0(e);
}
(C0.prototype = Object.create(Zi.prototype)).point = function(e, t) {
  Zi.prototype.point.call(this, t, e);
};
function M0(e) {
  this._context = e;
}
M0.prototype = {
  moveTo: function(e, t) {
    this._context.moveTo(t, e);
  },
  closePath: function() {
    this._context.closePath();
  },
  lineTo: function(e, t) {
    this._context.lineTo(t, e);
  },
  bezierCurveTo: function(e, t, r, n, i, a) {
    this._context.bezierCurveTo(t, e, n, r, a, i);
  }
};
function j1(e) {
  return new Zi(e);
}
function C1(e) {
  return new C0(e);
}
function I0(e) {
  this._context = e;
}
I0.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x = [], this._y = [];
  },
  lineEnd: function() {
    var e = this._x, t = this._y, r = e.length;
    if (r)
      if (this._line ? this._context.lineTo(e[0], t[0]) : this._context.moveTo(e[0], t[0]), r === 2)
        this._context.lineTo(e[1], t[1]);
      else
        for (var n = Mh(e), i = Mh(t), a = 0, o = 1; o < r; ++a, ++o)
          this._context.bezierCurveTo(n[0][a], i[0][a], n[1][a], i[1][a], e[o], t[o]);
    (this._line || this._line !== 0 && r === 1) && this._context.closePath(), this._line = 1 - this._line, this._x = this._y = null;
  },
  point: function(e, t) {
    this._x.push(+e), this._y.push(+t);
  }
};
function Mh(e) {
  var t, r = e.length - 1, n, i = new Array(r), a = new Array(r), o = new Array(r);
  for (i[0] = 0, a[0] = 2, o[0] = e[0] + 2 * e[1], t = 1; t < r - 1; ++t) i[t] = 1, a[t] = 4, o[t] = 4 * e[t] + 2 * e[t + 1];
  for (i[r - 1] = 2, a[r - 1] = 7, o[r - 1] = 8 * e[r - 1] + e[r], t = 1; t < r; ++t) n = i[t] / a[t - 1], a[t] -= n, o[t] -= n * o[t - 1];
  for (i[r - 1] = o[r - 1] / a[r - 1], t = r - 2; t >= 0; --t) i[t] = (o[t] - i[t + 1]) / a[t];
  for (a[r - 1] = (e[r] + i[r - 1]) / 2, t = 0; t < r - 1; ++t) a[t] = 2 * e[t + 1] - i[t + 1];
  return [i, a];
}
function M1(e) {
  return new I0(e);
}
function Za(e, t) {
  this._context = e, this._t = t;
}
Za.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x = this._y = NaN, this._point = 0;
  },
  lineEnd: function() {
    0 < this._t && this._t < 1 && this._point === 2 && this._context.lineTo(this._x, this._y), (this._line || this._line !== 0 && this._point === 1) && this._context.closePath(), this._line >= 0 && (this._t = 1 - this._t, this._line = 1 - this._line);
  },
  point: function(e, t) {
    switch (e = +e, t = +t, this._point) {
      case 0:
        this._point = 1, this._line ? this._context.lineTo(e, t) : this._context.moveTo(e, t);
        break;
      case 1:
        this._point = 2;
      // falls through
      default: {
        if (this._t <= 0)
          this._context.lineTo(this._x, t), this._context.lineTo(e, t);
        else {
          var r = this._x * (1 - this._t) + e * this._t;
          this._context.lineTo(r, this._y), this._context.lineTo(r, t);
        }
        break;
      }
    }
    this._x = e, this._y = t;
  }
};
function I1(e) {
  return new Za(e, 0.5);
}
function $1(e) {
  return new Za(e, 0);
}
function R1(e) {
  return new Za(e, 1);
}
function Nr(e, t) {
  if ((o = e.length) > 1)
    for (var r = 1, n, i, a = e[t[0]], o, u = a.length; r < o; ++r)
      for (i = a, a = e[t[r]], n = 0; n < u; ++n)
        a[n][1] += a[n][0] = isNaN(i[n][1]) ? i[n][0] : i[n][1];
}
function Nl(e) {
  for (var t = e.length, r = new Array(t); --t >= 0; ) r[t] = t;
  return r;
}
function k1(e, t) {
  return e[t];
}
function N1(e) {
  const t = [];
  return t.key = e, t;
}
function D1() {
  var e = Se([]), t = Nl, r = Nr, n = k1;
  function i(a) {
    var o = Array.from(e.apply(this, arguments), N1), u, s = o.length, c = -1, f;
    for (const l of a)
      for (u = 0, ++c; u < s; ++u)
        (o[u][c] = [0, +n(l, o[u].key, c, a)]).data = l;
    for (u = 0, f = ud(t(o)); u < s; ++u)
      o[f[u]].index = u;
    return r(o, f), o;
  }
  return i.keys = function(a) {
    return arguments.length ? (e = typeof a == "function" ? a : Se(Array.from(a)), i) : e;
  }, i.value = function(a) {
    return arguments.length ? (n = typeof a == "function" ? a : Se(+a), i) : n;
  }, i.order = function(a) {
    return arguments.length ? (t = a == null ? Nl : typeof a == "function" ? a : Se(Array.from(a)), i) : t;
  }, i.offset = function(a) {
    return arguments.length ? (r = a ?? Nr, i) : r;
  }, i;
}
function q1(e, t) {
  if ((n = e.length) > 0) {
    for (var r, n, i = 0, a = e[0].length, o; i < a; ++i) {
      for (o = r = 0; r < n; ++r) o += e[r][i][1] || 0;
      if (o) for (r = 0; r < n; ++r) e[r][i][1] /= o;
    }
    Nr(e, t);
  }
}
function L1(e, t) {
  if ((i = e.length) > 0) {
    for (var r = 0, n = e[t[0]], i, a = n.length; r < a; ++r) {
      for (var o = 0, u = 0; o < i; ++o) u += e[o][r][1] || 0;
      n[r][1] += n[r][0] = -u / 2;
    }
    Nr(e, t);
  }
}
function B1(e, t) {
  if (!(!((o = e.length) > 0) || !((a = (i = e[t[0]]).length) > 0))) {
    for (var r = 0, n = 1, i, a, o; n < a; ++n) {
      for (var u = 0, s = 0, c = 0; u < o; ++u) {
        for (var f = e[t[u]], l = f[n][1] || 0, d = f[n - 1][1] || 0, p = (l - d) / 2, y = 0; y < u; ++y) {
          var v = e[t[y]], h = v[n][1] || 0, g = v[n - 1][1] || 0;
          p += h - g;
        }
        s += l, c += p * l;
      }
      i[n - 1][1] += i[n - 1][0] = r, s && (r -= c / s);
    }
    i[n - 1][1] += i[n - 1][0] = r, Nr(e, t);
  }
}
function Nn(e) {
  "@babel/helpers - typeof";
  return Nn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Nn(e);
}
var F1 = ["type", "size", "sizeType"];
function Dl() {
  return Dl = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Dl.apply(this, arguments);
}
function Ih(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function $h(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ih(Object(r), !0).forEach(function(n) {
      z1(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ih(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function z1(e, t, r) {
  return t = U1(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function U1(e) {
  var t = W1(e, "string");
  return Nn(t) == "symbol" ? t : t + "";
}
function W1(e, t) {
  if (Nn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Nn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function G1(e, t) {
  if (e == null) return {};
  var r = H1(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function H1(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
var $0 = {
  symbolCircle: sd,
  symbolCross: p1,
  symbolDiamond: v1,
  symbolSquare: y1,
  symbolStar: x1,
  symbolTriangle: w1,
  symbolWye: _1
}, K1 = Math.PI / 180, V1 = function(t) {
  var r = "symbol".concat(Va(t));
  return $0[r] || sd;
}, Y1 = function(t, r, n) {
  if (r === "area")
    return t;
  switch (n) {
    case "cross":
      return 5 * t * t / 9;
    case "diamond":
      return 0.5 * t * t / Math.sqrt(3);
    case "square":
      return t * t;
    case "star": {
      var i = 18 * K1;
      return 1.25 * t * t * (Math.tan(i) - Math.tan(i * 2) * Math.pow(Math.tan(i), 2));
    }
    case "triangle":
      return Math.sqrt(3) * t * t / 4;
    case "wye":
      return (21 - 10 * Math.sqrt(3)) * t * t / 8;
    default:
      return Math.PI * t * t / 4;
  }
}, X1 = function(t, r) {
  $0["symbol".concat(Va(t))] = r;
}, cd = function(t) {
  var r = t.type, n = r === void 0 ? "circle" : r, i = t.size, a = i === void 0 ? 64 : i, o = t.sizeType, u = o === void 0 ? "area" : o, s = G1(t, F1), c = $h($h({}, s), {}, {
    type: n,
    size: a,
    sizeType: u
  }), f = function() {
    var h = V1(n), g = S1().type(h).size(Y1(a, u, n));
    return g();
  }, l = c.className, d = c.cx, p = c.cy, y = fe(c, !0);
  return d === +d && p === +p && a === +a ? /* @__PURE__ */ T.createElement("path", Dl({}, y, {
    className: pe("recharts-symbols", l),
    transform: "translate(".concat(d, ", ").concat(p, ")"),
    d: f()
  })) : null;
};
cd.registerSymbol = X1;
function Dr(e) {
  "@babel/helpers - typeof";
  return Dr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Dr(e);
}
function ql() {
  return ql = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, ql.apply(this, arguments);
}
function Rh(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Z1(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Rh(Object(r), !0).forEach(function(n) {
      Dn(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Rh(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function J1(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function Q1(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, k0(n.key), n);
  }
}
function eS(e, t, r) {
  return Q1(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function tS(e, t, r) {
  return t = Ji(t), rS(e, R0() ? Reflect.construct(t, r || [], Ji(e).constructor) : t.apply(e, r));
}
function rS(e, t) {
  if (t && (Dr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return nS(e);
}
function nS(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function R0() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (R0 = function() {
    return !!e;
  })();
}
function Ji(e) {
  return Ji = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Ji(e);
}
function iS(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Ll(e, t);
}
function Ll(e, t) {
  return Ll = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Ll(e, t);
}
function Dn(e, t, r) {
  return t = k0(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function k0(e) {
  var t = aS(e, "string");
  return Dr(t) == "symbol" ? t : t + "";
}
function aS(e, t) {
  if (Dr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Dr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var lt = 32, ld = /* @__PURE__ */ function(e) {
  function t() {
    return J1(this, t), tS(this, t, arguments);
  }
  return iS(t, e), eS(t, [{
    key: "renderIcon",
    value: (
      /**
       * Render the path of icon
       * @param {Object} data Data of each legend item
       * @return {String} Path element
       */
      function(n) {
        var i = this.props.inactiveColor, a = lt / 2, o = lt / 6, u = lt / 3, s = n.inactive ? i : n.color;
        if (n.type === "plainline")
          return /* @__PURE__ */ T.createElement("line", {
            strokeWidth: 4,
            fill: "none",
            stroke: s,
            strokeDasharray: n.payload.strokeDasharray,
            x1: 0,
            y1: a,
            x2: lt,
            y2: a,
            className: "recharts-legend-icon"
          });
        if (n.type === "line")
          return /* @__PURE__ */ T.createElement("path", {
            strokeWidth: 4,
            fill: "none",
            stroke: s,
            d: "M0,".concat(a, "h").concat(u, `
            A`).concat(o, ",").concat(o, ",0,1,1,").concat(2 * u, ",").concat(a, `
            H`).concat(lt, "M").concat(2 * u, ",").concat(a, `
            A`).concat(o, ",").concat(o, ",0,1,1,").concat(u, ",").concat(a),
            className: "recharts-legend-icon"
          });
        if (n.type === "rect")
          return /* @__PURE__ */ T.createElement("path", {
            stroke: "none",
            fill: s,
            d: "M0,".concat(lt / 8, "h").concat(lt, "v").concat(lt * 3 / 4, "h").concat(-32, "z"),
            className: "recharts-legend-icon"
          });
        if (/* @__PURE__ */ T.isValidElement(n.legendIcon)) {
          var c = Z1({}, n);
          return delete c.legendIcon, /* @__PURE__ */ T.cloneElement(n.legendIcon, c);
        }
        return /* @__PURE__ */ T.createElement(cd, {
          fill: s,
          cx: a,
          cy: a,
          size: lt,
          sizeType: "diameter",
          type: n.type
        });
      }
    )
    /**
     * Draw items of legend
     * @return {ReactElement} Items
     */
  }, {
    key: "renderItems",
    value: function() {
      var n = this, i = this.props, a = i.payload, o = i.iconSize, u = i.layout, s = i.formatter, c = i.inactiveColor, f = {
        x: 0,
        y: 0,
        width: lt,
        height: lt
      }, l = {
        display: u === "horizontal" ? "inline-block" : "block",
        marginRight: 10
      }, d = {
        display: "inline-block",
        verticalAlign: "middle",
        marginRight: 4
      };
      return a.map(function(p, y) {
        var v = p.formatter || s, h = pe(Dn(Dn({
          "recharts-legend-item": !0
        }, "legend-item-".concat(y), !0), "inactive", p.inactive));
        if (p.type === "none")
          return null;
        var g = ue(p.value) ? null : p.value;
        kr(
          !ue(p.value),
          `The name property is also required when using a function for the dataKey of a chart's cartesian components. Ex: <Bar name="Name of my Data"/>`
          // eslint-disable-line max-len
        );
        var w = p.inactive ? c : p.color;
        return /* @__PURE__ */ T.createElement("li", ql({
          className: h,
          style: l,
          key: "legend-item-".concat(y)
        }, Hi(n.props, p, y)), /* @__PURE__ */ T.createElement(Ml, {
          width: o,
          height: o,
          viewBox: f,
          style: d
        }, n.renderIcon(p)), /* @__PURE__ */ T.createElement("span", {
          className: "recharts-legend-item-text",
          style: {
            color: w
          }
        }, v ? v(g, p, y) : g));
      });
    }
  }, {
    key: "render",
    value: function() {
      var n = this.props, i = n.payload, a = n.layout, o = n.align;
      if (!i || !i.length)
        return null;
      var u = {
        padding: 0,
        margin: 0,
        textAlign: a === "horizontal" ? o : "left"
      };
      return /* @__PURE__ */ T.createElement("ul", {
        className: "recharts-default-legend",
        style: u
      }, this.renderItems());
    }
  }]);
}(Xt);
Dn(ld, "displayName", "Legend");
Dn(ld, "defaultProps", {
  iconSize: 14,
  layout: "horizontal",
  align: "center",
  verticalAlign: "middle",
  inactiveColor: "#ccc"
});
var qu, kh;
function oS() {
  if (kh) return qu;
  kh = 1;
  var e = Ga();
  function t() {
    this.__data__ = new e(), this.size = 0;
  }
  return qu = t, qu;
}
var Lu, Nh;
function uS() {
  if (Nh) return Lu;
  Nh = 1;
  function e(t) {
    var r = this.__data__, n = r.delete(t);
    return this.size = r.size, n;
  }
  return Lu = e, Lu;
}
var Bu, Dh;
function sS() {
  if (Dh) return Bu;
  Dh = 1;
  function e(t) {
    return this.__data__.get(t);
  }
  return Bu = e, Bu;
}
var Fu, qh;
function cS() {
  if (qh) return Fu;
  qh = 1;
  function e(t) {
    return this.__data__.has(t);
  }
  return Fu = e, Fu;
}
var zu, Lh;
function lS() {
  if (Lh) return zu;
  Lh = 1;
  var e = Ga(), t = ed(), r = td(), n = 200;
  function i(a, o) {
    var u = this.__data__;
    if (u instanceof e) {
      var s = u.__data__;
      if (!t || s.length < n - 1)
        return s.push([a, o]), this.size = ++u.size, this;
      u = this.__data__ = new r(s);
    }
    return u.set(a, o), this.size = u.size, this;
  }
  return zu = i, zu;
}
var Uu, Bh;
function N0() {
  if (Bh) return Uu;
  Bh = 1;
  var e = Ga(), t = oS(), r = uS(), n = sS(), i = cS(), a = lS();
  function o(u) {
    var s = this.__data__ = new e(u);
    this.size = s.size;
  }
  return o.prototype.clear = t, o.prototype.delete = r, o.prototype.get = n, o.prototype.has = i, o.prototype.set = a, Uu = o, Uu;
}
var Wu, Fh;
function fS() {
  if (Fh) return Wu;
  Fh = 1;
  var e = "__lodash_hash_undefined__";
  function t(r) {
    return this.__data__.set(r, e), this;
  }
  return Wu = t, Wu;
}
var Gu, zh;
function dS() {
  if (zh) return Gu;
  zh = 1;
  function e(t) {
    return this.__data__.has(t);
  }
  return Gu = e, Gu;
}
var Hu, Uh;
function D0() {
  if (Uh) return Hu;
  Uh = 1;
  var e = td(), t = fS(), r = dS();
  function n(i) {
    var a = -1, o = i == null ? 0 : i.length;
    for (this.__data__ = new e(); ++a < o; )
      this.add(i[a]);
  }
  return n.prototype.add = n.prototype.push = t, n.prototype.has = r, Hu = n, Hu;
}
var Ku, Wh;
function q0() {
  if (Wh) return Ku;
  Wh = 1;
  function e(t, r) {
    for (var n = -1, i = t == null ? 0 : t.length; ++n < i; )
      if (r(t[n], n, t))
        return !0;
    return !1;
  }
  return Ku = e, Ku;
}
var Vu, Gh;
function L0() {
  if (Gh) return Vu;
  Gh = 1;
  function e(t, r) {
    return t.has(r);
  }
  return Vu = e, Vu;
}
var Yu, Hh;
function B0() {
  if (Hh) return Yu;
  Hh = 1;
  var e = D0(), t = q0(), r = L0(), n = 1, i = 2;
  function a(o, u, s, c, f, l) {
    var d = s & n, p = o.length, y = u.length;
    if (p != y && !(d && y > p))
      return !1;
    var v = l.get(o), h = l.get(u);
    if (v && h)
      return v == u && h == o;
    var g = -1, w = !0, b = s & i ? new e() : void 0;
    for (l.set(o, u), l.set(u, o); ++g < p; ) {
      var O = o[g], m = u[g];
      if (c)
        var x = d ? c(m, O, g, u, o, l) : c(O, m, g, o, u, l);
      if (x !== void 0) {
        if (x)
          continue;
        w = !1;
        break;
      }
      if (b) {
        if (!t(u, function(_, P) {
          if (!r(b, P) && (O === _ || f(O, _, s, c, l)))
            return b.push(P);
        })) {
          w = !1;
          break;
        }
      } else if (!(O === m || f(O, m, s, c, l))) {
        w = !1;
        break;
      }
    }
    return l.delete(o), l.delete(u), w;
  }
  return Yu = a, Yu;
}
var Xu, Kh;
function pS() {
  if (Kh) return Xu;
  Kh = 1;
  var e = St(), t = e.Uint8Array;
  return Xu = t, Xu;
}
var Zu, Vh;
function hS() {
  if (Vh) return Zu;
  Vh = 1;
  function e(t) {
    var r = -1, n = Array(t.size);
    return t.forEach(function(i, a) {
      n[++r] = [a, i];
    }), n;
  }
  return Zu = e, Zu;
}
var Ju, Yh;
function fd() {
  if (Yh) return Ju;
  Yh = 1;
  function e(t) {
    var r = -1, n = Array(t.size);
    return t.forEach(function(i) {
      n[++r] = i;
    }), n;
  }
  return Ju = e, Ju;
}
var Qu, Xh;
function vS() {
  if (Xh) return Qu;
  Xh = 1;
  var e = vi(), t = pS(), r = Qf(), n = B0(), i = hS(), a = fd(), o = 1, u = 2, s = "[object Boolean]", c = "[object Date]", f = "[object Error]", l = "[object Map]", d = "[object Number]", p = "[object RegExp]", y = "[object Set]", v = "[object String]", h = "[object Symbol]", g = "[object ArrayBuffer]", w = "[object DataView]", b = e ? e.prototype : void 0, O = b ? b.valueOf : void 0;
  function m(x, _, P, E, I, S, j) {
    switch (P) {
      case w:
        if (x.byteLength != _.byteLength || x.byteOffset != _.byteOffset)
          return !1;
        x = x.buffer, _ = _.buffer;
      case g:
        return !(x.byteLength != _.byteLength || !S(new t(x), new t(_)));
      case s:
      case c:
      case d:
        return r(+x, +_);
      case f:
        return x.name == _.name && x.message == _.message;
      case p:
      case v:
        return x == _ + "";
      case l:
        var M = i;
      case y:
        var R = E & o;
        if (M || (M = a), x.size != _.size && !R)
          return !1;
        var k = j.get(x);
        if (k)
          return k == _;
        E |= u, j.set(x, _);
        var q = n(M(x), M(_), E, I, S, j);
        return j.delete(x), q;
      case h:
        if (O)
          return O.call(x) == O.call(_);
    }
    return !1;
  }
  return Qu = m, Qu;
}
var es, Zh;
function F0() {
  if (Zh) return es;
  Zh = 1;
  function e(t, r) {
    for (var n = -1, i = r.length, a = t.length; ++n < i; )
      t[a + n] = r[n];
    return t;
  }
  return es = e, es;
}
var ts, Jh;
function yS() {
  if (Jh) return ts;
  Jh = 1;
  var e = F0(), t = Xe();
  function r(n, i, a) {
    var o = i(n);
    return t(n) ? o : e(o, a(n));
  }
  return ts = r, ts;
}
var rs, Qh;
function mS() {
  if (Qh) return rs;
  Qh = 1;
  function e(t, r) {
    for (var n = -1, i = t == null ? 0 : t.length, a = 0, o = []; ++n < i; ) {
      var u = t[n];
      r(u, n, t) && (o[a++] = u);
    }
    return o;
  }
  return rs = e, rs;
}
var ns, ev;
function gS() {
  if (ev) return ns;
  ev = 1;
  function e() {
    return [];
  }
  return ns = e, ns;
}
var is, tv;
function bS() {
  if (tv) return is;
  tv = 1;
  var e = mS(), t = gS(), r = Object.prototype, n = r.propertyIsEnumerable, i = Object.getOwnPropertySymbols, a = i ? function(o) {
    return o == null ? [] : (o = Object(o), e(i(o), function(u) {
      return n.call(o, u);
    }));
  } : t;
  return is = a, is;
}
var as, rv;
function xS() {
  if (rv) return as;
  rv = 1;
  function e(t, r) {
    for (var n = -1, i = Array(t); ++n < t; )
      i[n] = r(n);
    return i;
  }
  return as = e, as;
}
var os, nv;
function wS() {
  if (nv) return os;
  nv = 1;
  var e = Lt(), t = Bt(), r = "[object Arguments]";
  function n(i) {
    return t(i) && e(i) == r;
  }
  return os = n, os;
}
var us, iv;
function dd() {
  if (iv) return us;
  iv = 1;
  var e = wS(), t = Bt(), r = Object.prototype, n = r.hasOwnProperty, i = r.propertyIsEnumerable, a = e(/* @__PURE__ */ function() {
    return arguments;
  }()) ? e : function(o) {
    return t(o) && n.call(o, "callee") && !i.call(o, "callee");
  };
  return us = a, us;
}
var Tn = { exports: {} }, ss, av;
function OS() {
  if (av) return ss;
  av = 1;
  function e() {
    return !1;
  }
  return ss = e, ss;
}
Tn.exports;
var ov;
function z0() {
  return ov || (ov = 1, function(e, t) {
    var r = St(), n = OS(), i = t && !t.nodeType && t, a = i && !0 && e && !e.nodeType && e, o = a && a.exports === i, u = o ? r.Buffer : void 0, s = u ? u.isBuffer : void 0, c = s || n;
    e.exports = c;
  }(Tn, Tn.exports)), Tn.exports;
}
var cs, uv;
function pd() {
  if (uv) return cs;
  uv = 1;
  var e = 9007199254740991, t = /^(?:0|[1-9]\d*)$/;
  function r(n, i) {
    var a = typeof n;
    return i = i ?? e, !!i && (a == "number" || a != "symbol" && t.test(n)) && n > -1 && n % 1 == 0 && n < i;
  }
  return cs = r, cs;
}
var ls, sv;
function hd() {
  if (sv) return ls;
  sv = 1;
  var e = 9007199254740991;
  function t(r) {
    return typeof r == "number" && r > -1 && r % 1 == 0 && r <= e;
  }
  return ls = t, ls;
}
var fs, cv;
function _S() {
  if (cv) return fs;
  cv = 1;
  var e = Lt(), t = hd(), r = Bt(), n = "[object Arguments]", i = "[object Array]", a = "[object Boolean]", o = "[object Date]", u = "[object Error]", s = "[object Function]", c = "[object Map]", f = "[object Number]", l = "[object Object]", d = "[object RegExp]", p = "[object Set]", y = "[object String]", v = "[object WeakMap]", h = "[object ArrayBuffer]", g = "[object DataView]", w = "[object Float32Array]", b = "[object Float64Array]", O = "[object Int8Array]", m = "[object Int16Array]", x = "[object Int32Array]", _ = "[object Uint8Array]", P = "[object Uint8ClampedArray]", E = "[object Uint16Array]", I = "[object Uint32Array]", S = {};
  S[w] = S[b] = S[O] = S[m] = S[x] = S[_] = S[P] = S[E] = S[I] = !0, S[n] = S[i] = S[h] = S[a] = S[g] = S[o] = S[u] = S[s] = S[c] = S[f] = S[l] = S[d] = S[p] = S[y] = S[v] = !1;
  function j(M) {
    return r(M) && t(M.length) && !!S[e(M)];
  }
  return fs = j, fs;
}
var ds, lv;
function U0() {
  if (lv) return ds;
  lv = 1;
  function e(t) {
    return function(r) {
      return t(r);
    };
  }
  return ds = e, ds;
}
var jn = { exports: {} };
jn.exports;
var fv;
function SS() {
  return fv || (fv = 1, function(e, t) {
    var r = c0(), n = t && !t.nodeType && t, i = n && !0 && e && !e.nodeType && e, a = i && i.exports === n, o = a && r.process, u = function() {
      try {
        var s = i && i.require && i.require("util").types;
        return s || o && o.binding && o.binding("util");
      } catch {
      }
    }();
    e.exports = u;
  }(jn, jn.exports)), jn.exports;
}
var ps, dv;
function W0() {
  if (dv) return ps;
  dv = 1;
  var e = _S(), t = U0(), r = SS(), n = r && r.isTypedArray, i = n ? t(n) : e;
  return ps = i, ps;
}
var hs, pv;
function PS() {
  if (pv) return hs;
  pv = 1;
  var e = xS(), t = dd(), r = Xe(), n = z0(), i = pd(), a = W0(), o = Object.prototype, u = o.hasOwnProperty;
  function s(c, f) {
    var l = r(c), d = !l && t(c), p = !l && !d && n(c), y = !l && !d && !p && a(c), v = l || d || p || y, h = v ? e(c.length, String) : [], g = h.length;
    for (var w in c)
      (f || u.call(c, w)) && !(v && // Safari 9 has enumerable `arguments.length` in strict mode.
      (w == "length" || // Node.js 0.10 has enumerable non-index properties on buffers.
      p && (w == "offset" || w == "parent") || // PhantomJS 2 has enumerable non-index properties on typed arrays.
      y && (w == "buffer" || w == "byteLength" || w == "byteOffset") || // Skip index properties.
      i(w, g))) && h.push(w);
    return h;
  }
  return hs = s, hs;
}
var vs, hv;
function AS() {
  if (hv) return vs;
  hv = 1;
  var e = Object.prototype;
  function t(r) {
    var n = r && r.constructor, i = typeof n == "function" && n.prototype || e;
    return r === i;
  }
  return vs = t, vs;
}
var ys, vv;
function G0() {
  if (vv) return ys;
  vv = 1;
  function e(t, r) {
    return function(n) {
      return t(r(n));
    };
  }
  return ys = e, ys;
}
var ms, yv;
function ES() {
  if (yv) return ms;
  yv = 1;
  var e = G0(), t = e(Object.keys, Object);
  return ms = t, ms;
}
var gs, mv;
function TS() {
  if (mv) return gs;
  mv = 1;
  var e = AS(), t = ES(), r = Object.prototype, n = r.hasOwnProperty;
  function i(a) {
    if (!e(a))
      return t(a);
    var o = [];
    for (var u in Object(a))
      n.call(a, u) && u != "constructor" && o.push(u);
    return o;
  }
  return gs = i, gs;
}
var bs, gv;
function bi() {
  if (gv) return bs;
  gv = 1;
  var e = Jf(), t = hd();
  function r(n) {
    return n != null && t(n.length) && !e(n);
  }
  return bs = r, bs;
}
var xs, bv;
function Ja() {
  if (bv) return xs;
  bv = 1;
  var e = PS(), t = TS(), r = bi();
  function n(i) {
    return r(i) ? e(i) : t(i);
  }
  return xs = n, xs;
}
var ws, xv;
function jS() {
  if (xv) return ws;
  xv = 1;
  var e = yS(), t = bS(), r = Ja();
  function n(i) {
    return e(i, r, t);
  }
  return ws = n, ws;
}
var Os, wv;
function CS() {
  if (wv) return Os;
  wv = 1;
  var e = jS(), t = 1, r = Object.prototype, n = r.hasOwnProperty;
  function i(a, o, u, s, c, f) {
    var l = u & t, d = e(a), p = d.length, y = e(o), v = y.length;
    if (p != v && !l)
      return !1;
    for (var h = p; h--; ) {
      var g = d[h];
      if (!(l ? g in o : n.call(o, g)))
        return !1;
    }
    var w = f.get(a), b = f.get(o);
    if (w && b)
      return w == o && b == a;
    var O = !0;
    f.set(a, o), f.set(o, a);
    for (var m = l; ++h < p; ) {
      g = d[h];
      var x = a[g], _ = o[g];
      if (s)
        var P = l ? s(_, x, g, o, a, f) : s(x, _, g, a, o, f);
      if (!(P === void 0 ? x === _ || c(x, _, u, s, f) : P)) {
        O = !1;
        break;
      }
      m || (m = g == "constructor");
    }
    if (O && !m) {
      var E = a.constructor, I = o.constructor;
      E != I && "constructor" in a && "constructor" in o && !(typeof E == "function" && E instanceof E && typeof I == "function" && I instanceof I) && (O = !1);
    }
    return f.delete(a), f.delete(o), O;
  }
  return Os = i, Os;
}
var _s, Ov;
function MS() {
  if (Ov) return _s;
  Ov = 1;
  var e = gr(), t = St(), r = e(t, "DataView");
  return _s = r, _s;
}
var Ss, _v;
function IS() {
  if (_v) return Ss;
  _v = 1;
  var e = gr(), t = St(), r = e(t, "Promise");
  return Ss = r, Ss;
}
var Ps, Sv;
function H0() {
  if (Sv) return Ps;
  Sv = 1;
  var e = gr(), t = St(), r = e(t, "Set");
  return Ps = r, Ps;
}
var As, Pv;
function $S() {
  if (Pv) return As;
  Pv = 1;
  var e = gr(), t = St(), r = e(t, "WeakMap");
  return As = r, As;
}
var Es, Av;
function RS() {
  if (Av) return Es;
  Av = 1;
  var e = MS(), t = ed(), r = IS(), n = H0(), i = $S(), a = Lt(), o = l0(), u = "[object Map]", s = "[object Object]", c = "[object Promise]", f = "[object Set]", l = "[object WeakMap]", d = "[object DataView]", p = o(e), y = o(t), v = o(r), h = o(n), g = o(i), w = a;
  return (e && w(new e(new ArrayBuffer(1))) != d || t && w(new t()) != u || r && w(r.resolve()) != c || n && w(new n()) != f || i && w(new i()) != l) && (w = function(b) {
    var O = a(b), m = O == s ? b.constructor : void 0, x = m ? o(m) : "";
    if (x)
      switch (x) {
        case p:
          return d;
        case y:
          return u;
        case v:
          return c;
        case h:
          return f;
        case g:
          return l;
      }
    return O;
  }), Es = w, Es;
}
var Ts, Ev;
function kS() {
  if (Ev) return Ts;
  Ev = 1;
  var e = N0(), t = B0(), r = vS(), n = CS(), i = RS(), a = Xe(), o = z0(), u = W0(), s = 1, c = "[object Arguments]", f = "[object Array]", l = "[object Object]", d = Object.prototype, p = d.hasOwnProperty;
  function y(v, h, g, w, b, O) {
    var m = a(v), x = a(h), _ = m ? f : i(v), P = x ? f : i(h);
    _ = _ == c ? l : _, P = P == c ? l : P;
    var E = _ == l, I = P == l, S = _ == P;
    if (S && o(v)) {
      if (!o(h))
        return !1;
      m = !0, E = !1;
    }
    if (S && !E)
      return O || (O = new e()), m || u(v) ? t(v, h, g, w, b, O) : r(v, h, _, g, w, b, O);
    if (!(g & s)) {
      var j = E && p.call(v, "__wrapped__"), M = I && p.call(h, "__wrapped__");
      if (j || M) {
        var R = j ? v.value() : v, k = M ? h.value() : h;
        return O || (O = new e()), b(R, k, g, w, O);
      }
    }
    return S ? (O || (O = new e()), n(v, h, g, w, b, O)) : !1;
  }
  return Ts = y, Ts;
}
var js, Tv;
function vd() {
  if (Tv) return js;
  Tv = 1;
  var e = kS(), t = Bt();
  function r(n, i, a, o, u) {
    return n === i ? !0 : n == null || i == null || !t(n) && !t(i) ? n !== n && i !== i : e(n, i, a, o, r, u);
  }
  return js = r, js;
}
var Cs, jv;
function NS() {
  if (jv) return Cs;
  jv = 1;
  var e = N0(), t = vd(), r = 1, n = 2;
  function i(a, o, u, s) {
    var c = u.length, f = c, l = !s;
    if (a == null)
      return !f;
    for (a = Object(a); c--; ) {
      var d = u[c];
      if (l && d[2] ? d[1] !== a[d[0]] : !(d[0] in a))
        return !1;
    }
    for (; ++c < f; ) {
      d = u[c];
      var p = d[0], y = a[p], v = d[1];
      if (l && d[2]) {
        if (y === void 0 && !(p in a))
          return !1;
      } else {
        var h = new e();
        if (s)
          var g = s(y, v, p, a, o, h);
        if (!(g === void 0 ? t(v, y, r | n, s, h) : g))
          return !1;
      }
    }
    return !0;
  }
  return Cs = i, Cs;
}
var Ms, Cv;
function K0() {
  if (Cv) return Ms;
  Cv = 1;
  var e = Zt();
  function t(r) {
    return r === r && !e(r);
  }
  return Ms = t, Ms;
}
var Is, Mv;
function DS() {
  if (Mv) return Is;
  Mv = 1;
  var e = K0(), t = Ja();
  function r(n) {
    for (var i = t(n), a = i.length; a--; ) {
      var o = i[a], u = n[o];
      i[a] = [o, u, e(u)];
    }
    return i;
  }
  return Is = r, Is;
}
var $s, Iv;
function V0() {
  if (Iv) return $s;
  Iv = 1;
  function e(t, r) {
    return function(n) {
      return n == null ? !1 : n[t] === r && (r !== void 0 || t in Object(n));
    };
  }
  return $s = e, $s;
}
var Rs, $v;
function qS() {
  if ($v) return Rs;
  $v = 1;
  var e = NS(), t = DS(), r = V0();
  function n(i) {
    var a = t(i);
    return a.length == 1 && a[0][2] ? r(a[0][0], a[0][1]) : function(o) {
      return o === i || e(o, i, a);
    };
  }
  return Rs = n, Rs;
}
var ks, Rv;
function LS() {
  if (Rv) return ks;
  Rv = 1;
  function e(t, r) {
    return t != null && r in Object(t);
  }
  return ks = e, ks;
}
var Ns, kv;
function BS() {
  if (kv) return Ns;
  kv = 1;
  var e = p0(), t = dd(), r = Xe(), n = pd(), i = hd(), a = Ka();
  function o(u, s, c) {
    s = e(s, u);
    for (var f = -1, l = s.length, d = !1; ++f < l; ) {
      var p = a(s[f]);
      if (!(d = u != null && c(u, p)))
        break;
      u = u[p];
    }
    return d || ++f != l ? d : (l = u == null ? 0 : u.length, !!l && i(l) && n(p, l) && (r(u) || t(u)));
  }
  return Ns = o, Ns;
}
var Ds, Nv;
function FS() {
  if (Nv) return Ds;
  Nv = 1;
  var e = LS(), t = BS();
  function r(n, i) {
    return n != null && t(n, i, e);
  }
  return Ds = r, Ds;
}
var qs, Dv;
function zS() {
  if (Dv) return qs;
  Dv = 1;
  var e = vd(), t = h0(), r = FS(), n = Zf(), i = K0(), a = V0(), o = Ka(), u = 1, s = 2;
  function c(f, l) {
    return n(f) && i(l) ? a(o(f), l) : function(d) {
      var p = t(d, f);
      return p === void 0 && p === l ? r(d, f) : e(l, p, u | s);
    };
  }
  return qs = c, qs;
}
var Ls, qv;
function cn() {
  if (qv) return Ls;
  qv = 1;
  function e(t) {
    return t;
  }
  return Ls = e, Ls;
}
var Bs, Lv;
function US() {
  if (Lv) return Bs;
  Lv = 1;
  function e(t) {
    return function(r) {
      return r == null ? void 0 : r[t];
    };
  }
  return Bs = e, Bs;
}
var Fs, Bv;
function WS() {
  if (Bv) return Fs;
  Bv = 1;
  var e = nd();
  function t(r) {
    return function(n) {
      return e(n, r);
    };
  }
  return Fs = t, Fs;
}
var zs, Fv;
function GS() {
  if (Fv) return zs;
  Fv = 1;
  var e = US(), t = WS(), r = Zf(), n = Ka();
  function i(a) {
    return r(a) ? e(n(a)) : t(a);
  }
  return zs = i, zs;
}
var Us, zv;
function Jt() {
  if (zv) return Us;
  zv = 1;
  var e = qS(), t = zS(), r = cn(), n = Xe(), i = GS();
  function a(o) {
    return typeof o == "function" ? o : o == null ? r : typeof o == "object" ? n(o) ? t(o[0], o[1]) : e(o) : i(o);
  }
  return Us = a, Us;
}
var Ws, Uv;
function Y0() {
  if (Uv) return Ws;
  Uv = 1;
  function e(t, r, n, i) {
    for (var a = t.length, o = n + (i ? 1 : -1); i ? o-- : ++o < a; )
      if (r(t[o], o, t))
        return o;
    return -1;
  }
  return Ws = e, Ws;
}
var Gs, Wv;
function HS() {
  if (Wv) return Gs;
  Wv = 1;
  function e(t) {
    return t !== t;
  }
  return Gs = e, Gs;
}
var Hs, Gv;
function KS() {
  if (Gv) return Hs;
  Gv = 1;
  function e(t, r, n) {
    for (var i = n - 1, a = t.length; ++i < a; )
      if (t[i] === r)
        return i;
    return -1;
  }
  return Hs = e, Hs;
}
var Ks, Hv;
function VS() {
  if (Hv) return Ks;
  Hv = 1;
  var e = Y0(), t = HS(), r = KS();
  function n(i, a, o) {
    return a === a ? r(i, a, o) : e(i, t, o);
  }
  return Ks = n, Ks;
}
var Vs, Kv;
function YS() {
  if (Kv) return Vs;
  Kv = 1;
  var e = VS();
  function t(r, n) {
    var i = r == null ? 0 : r.length;
    return !!i && e(r, n, 0) > -1;
  }
  return Vs = t, Vs;
}
var Ys, Vv;
function XS() {
  if (Vv) return Ys;
  Vv = 1;
  function e(t, r, n) {
    for (var i = -1, a = t == null ? 0 : t.length; ++i < a; )
      if (n(r, t[i]))
        return !0;
    return !1;
  }
  return Ys = e, Ys;
}
var Xs, Yv;
function ZS() {
  if (Yv) return Xs;
  Yv = 1;
  function e() {
  }
  return Xs = e, Xs;
}
var Zs, Xv;
function JS() {
  if (Xv) return Zs;
  Xv = 1;
  var e = H0(), t = ZS(), r = fd(), n = 1 / 0, i = e && 1 / r(new e([, -0]))[1] == n ? function(a) {
    return new e(a);
  } : t;
  return Zs = i, Zs;
}
var Js, Zv;
function QS() {
  if (Zv) return Js;
  Zv = 1;
  var e = D0(), t = YS(), r = XS(), n = L0(), i = JS(), a = fd(), o = 200;
  function u(s, c, f) {
    var l = -1, d = t, p = s.length, y = !0, v = [], h = v;
    if (f)
      y = !1, d = r;
    else if (p >= o) {
      var g = c ? null : i(s);
      if (g)
        return a(g);
      y = !1, d = n, h = new e();
    } else
      h = c ? [] : v;
    e:
      for (; ++l < p; ) {
        var w = s[l], b = c ? c(w) : w;
        if (w = f || w !== 0 ? w : 0, y && b === b) {
          for (var O = h.length; O--; )
            if (h[O] === b)
              continue e;
          c && h.push(b), v.push(w);
        } else d(h, b, f) || (h !== v && h.push(b), v.push(w));
      }
    return v;
  }
  return Js = u, Js;
}
var Qs, Jv;
function eP() {
  if (Jv) return Qs;
  Jv = 1;
  var e = Jt(), t = QS();
  function r(n, i) {
    return n && n.length ? t(n, e(i, 2)) : [];
  }
  return Qs = r, Qs;
}
var tP = eP();
const Qv = /* @__PURE__ */ Pe(tP);
function X0(e, t, r) {
  return t === !0 ? Qv(e, r) : ue(t) ? Qv(e, t) : e;
}
function qr(e) {
  "@babel/helpers - typeof";
  return qr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, qr(e);
}
var rP = ["ref"];
function ey(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Tt(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ey(Object(r), !0).forEach(function(n) {
      Qa(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ey(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function nP(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function ty(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, J0(n.key), n);
  }
}
function iP(e, t, r) {
  return ty(e.prototype, t), ty(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function aP(e, t, r) {
  return t = Qi(t), oP(e, Z0() ? Reflect.construct(t, r, Qi(e).constructor) : t.apply(e, r));
}
function oP(e, t) {
  if (t && (qr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return uP(e);
}
function uP(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function Z0() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (Z0 = function() {
    return !!e;
  })();
}
function Qi(e) {
  return Qi = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Qi(e);
}
function sP(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Bl(e, t);
}
function Bl(e, t) {
  return Bl = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Bl(e, t);
}
function Qa(e, t, r) {
  return t = J0(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function J0(e) {
  var t = cP(e, "string");
  return qr(t) == "symbol" ? t : t + "";
}
function cP(e, t) {
  if (qr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (qr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function lP(e, t) {
  if (e == null) return {};
  var r = fP(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function fP(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function dP(e) {
  return e.value;
}
function pP(e, t) {
  if (/* @__PURE__ */ T.isValidElement(e))
    return /* @__PURE__ */ T.cloneElement(e, t);
  if (typeof e == "function")
    return /* @__PURE__ */ T.createElement(e, t);
  t.ref;
  var r = lP(t, rP);
  return /* @__PURE__ */ T.createElement(ld, r);
}
var ry = 1, Ir = /* @__PURE__ */ function(e) {
  function t() {
    var r;
    nP(this, t);
    for (var n = arguments.length, i = new Array(n), a = 0; a < n; a++)
      i[a] = arguments[a];
    return r = aP(this, t, [].concat(i)), Qa(r, "lastBoundingBox", {
      width: -1,
      height: -1
    }), r;
  }
  return sP(t, e), iP(t, [{
    key: "componentDidMount",
    value: function() {
      this.updateBBox();
    }
  }, {
    key: "componentDidUpdate",
    value: function() {
      this.updateBBox();
    }
  }, {
    key: "getBBox",
    value: function() {
      if (this.wrapperNode && this.wrapperNode.getBoundingClientRect) {
        var n = this.wrapperNode.getBoundingClientRect();
        return n.height = this.wrapperNode.offsetHeight, n.width = this.wrapperNode.offsetWidth, n;
      }
      return null;
    }
  }, {
    key: "updateBBox",
    value: function() {
      var n = this.props.onBBoxUpdate, i = this.getBBox();
      i ? (Math.abs(i.width - this.lastBoundingBox.width) > ry || Math.abs(i.height - this.lastBoundingBox.height) > ry) && (this.lastBoundingBox.width = i.width, this.lastBoundingBox.height = i.height, n && n(i)) : (this.lastBoundingBox.width !== -1 || this.lastBoundingBox.height !== -1) && (this.lastBoundingBox.width = -1, this.lastBoundingBox.height = -1, n && n(null));
    }
  }, {
    key: "getBBoxSnapshot",
    value: function() {
      return this.lastBoundingBox.width >= 0 && this.lastBoundingBox.height >= 0 ? Tt({}, this.lastBoundingBox) : {
        width: 0,
        height: 0
      };
    }
  }, {
    key: "getDefaultPosition",
    value: function(n) {
      var i = this.props, a = i.layout, o = i.align, u = i.verticalAlign, s = i.margin, c = i.chartWidth, f = i.chartHeight, l, d;
      if (!n || (n.left === void 0 || n.left === null) && (n.right === void 0 || n.right === null))
        if (o === "center" && a === "vertical") {
          var p = this.getBBoxSnapshot();
          l = {
            left: ((c || 0) - p.width) / 2
          };
        } else
          l = o === "right" ? {
            right: s && s.right || 0
          } : {
            left: s && s.left || 0
          };
      if (!n || (n.top === void 0 || n.top === null) && (n.bottom === void 0 || n.bottom === null))
        if (u === "middle") {
          var y = this.getBBoxSnapshot();
          d = {
            top: ((f || 0) - y.height) / 2
          };
        } else
          d = u === "bottom" ? {
            bottom: s && s.bottom || 0
          } : {
            top: s && s.top || 0
          };
      return Tt(Tt({}, l), d);
    }
  }, {
    key: "render",
    value: function() {
      var n = this, i = this.props, a = i.content, o = i.width, u = i.height, s = i.wrapperStyle, c = i.payloadUniqBy, f = i.payload, l = Tt(Tt({
        position: "absolute",
        width: o || "auto",
        height: u || "auto"
      }, this.getDefaultPosition(s)), s);
      return /* @__PURE__ */ T.createElement("div", {
        className: "recharts-legend-wrapper",
        style: l,
        ref: function(p) {
          n.wrapperNode = p;
        }
      }, pP(a, Tt(Tt({}, this.props), {}, {
        payload: X0(f, c, dP)
      })));
    }
  }], [{
    key: "getWithHeight",
    value: function(n, i) {
      var a = Tt(Tt({}, this.defaultProps), n.props), o = a.layout;
      return o === "vertical" && K(n.props.height) ? {
        height: n.props.height
      } : o === "horizontal" ? {
        width: n.props.width || i
      } : null;
    }
  }]);
}(Xt);
Qa(Ir, "displayName", "Legend");
Qa(Ir, "defaultProps", {
  iconSize: 14,
  layout: "horizontal",
  align: "center",
  verticalAlign: "bottom"
});
var ec, ny;
function hP() {
  if (ny) return ec;
  ny = 1;
  var e = vi(), t = dd(), r = Xe(), n = e ? e.isConcatSpreadable : void 0;
  function i(a) {
    return r(a) || t(a) || !!(n && a && a[n]);
  }
  return ec = i, ec;
}
var tc, iy;
function Q0() {
  if (iy) return tc;
  iy = 1;
  var e = F0(), t = hP();
  function r(n, i, a, o, u) {
    var s = -1, c = n.length;
    for (a || (a = t), u || (u = []); ++s < c; ) {
      var f = n[s];
      i > 0 && a(f) ? i > 1 ? r(f, i - 1, a, o, u) : e(u, f) : o || (u[u.length] = f);
    }
    return u;
  }
  return tc = r, tc;
}
var rc, ay;
function vP() {
  if (ay) return rc;
  ay = 1;
  function e(t) {
    return function(r, n, i) {
      for (var a = -1, o = Object(r), u = i(r), s = u.length; s--; ) {
        var c = u[t ? s : ++a];
        if (n(o[c], c, o) === !1)
          break;
      }
      return r;
    };
  }
  return rc = e, rc;
}
var nc, oy;
function yP() {
  if (oy) return nc;
  oy = 1;
  var e = vP(), t = e();
  return nc = t, nc;
}
var ic, uy;
function ex() {
  if (uy) return ic;
  uy = 1;
  var e = yP(), t = Ja();
  function r(n, i) {
    return n && e(n, i, t);
  }
  return ic = r, ic;
}
var ac, sy;
function mP() {
  if (sy) return ac;
  sy = 1;
  var e = bi();
  function t(r, n) {
    return function(i, a) {
      if (i == null)
        return i;
      if (!e(i))
        return r(i, a);
      for (var o = i.length, u = n ? o : -1, s = Object(i); (n ? u-- : ++u < o) && a(s[u], u, s) !== !1; )
        ;
      return i;
    };
  }
  return ac = t, ac;
}
var oc, cy;
function yd() {
  if (cy) return oc;
  cy = 1;
  var e = ex(), t = mP(), r = t(e);
  return oc = r, oc;
}
var uc, ly;
function tx() {
  if (ly) return uc;
  ly = 1;
  var e = yd(), t = bi();
  function r(n, i) {
    var a = -1, o = t(n) ? Array(n.length) : [];
    return e(n, function(u, s, c) {
      o[++a] = i(u, s, c);
    }), o;
  }
  return uc = r, uc;
}
var sc, fy;
function gP() {
  if (fy) return sc;
  fy = 1;
  function e(t, r) {
    var n = t.length;
    for (t.sort(r); n--; )
      t[n] = t[n].value;
    return t;
  }
  return sc = e, sc;
}
var cc, dy;
function bP() {
  if (dy) return cc;
  dy = 1;
  var e = un();
  function t(r, n) {
    if (r !== n) {
      var i = r !== void 0, a = r === null, o = r === r, u = e(r), s = n !== void 0, c = n === null, f = n === n, l = e(n);
      if (!c && !l && !u && r > n || u && s && f && !c && !l || a && s && f || !i && f || !o)
        return 1;
      if (!a && !u && !l && r < n || l && i && o && !a && !u || c && i && o || !s && o || !f)
        return -1;
    }
    return 0;
  }
  return cc = t, cc;
}
var lc, py;
function xP() {
  if (py) return lc;
  py = 1;
  var e = bP();
  function t(r, n, i) {
    for (var a = -1, o = r.criteria, u = n.criteria, s = o.length, c = i.length; ++a < s; ) {
      var f = e(o[a], u[a]);
      if (f) {
        if (a >= c)
          return f;
        var l = i[a];
        return f * (l == "desc" ? -1 : 1);
      }
    }
    return r.index - n.index;
  }
  return lc = t, lc;
}
var fc, hy;
function wP() {
  if (hy) return fc;
  hy = 1;
  var e = rd(), t = nd(), r = Jt(), n = tx(), i = gP(), a = U0(), o = xP(), u = cn(), s = Xe();
  function c(f, l, d) {
    l.length ? l = e(l, function(v) {
      return s(v) ? function(h) {
        return t(h, v.length === 1 ? v[0] : v);
      } : v;
    }) : l = [u];
    var p = -1;
    l = e(l, a(r));
    var y = n(f, function(v, h, g) {
      var w = e(l, function(b) {
        return b(v);
      });
      return { criteria: w, index: ++p, value: v };
    });
    return i(y, function(v, h) {
      return o(v, h, d);
    });
  }
  return fc = c, fc;
}
var dc, vy;
function OP() {
  if (vy) return dc;
  vy = 1;
  function e(t, r, n) {
    switch (n.length) {
      case 0:
        return t.call(r);
      case 1:
        return t.call(r, n[0]);
      case 2:
        return t.call(r, n[0], n[1]);
      case 3:
        return t.call(r, n[0], n[1], n[2]);
    }
    return t.apply(r, n);
  }
  return dc = e, dc;
}
var pc, yy;
function _P() {
  if (yy) return pc;
  yy = 1;
  var e = OP(), t = Math.max;
  function r(n, i, a) {
    return i = t(i === void 0 ? n.length - 1 : i, 0), function() {
      for (var o = arguments, u = -1, s = t(o.length - i, 0), c = Array(s); ++u < s; )
        c[u] = o[i + u];
      u = -1;
      for (var f = Array(i + 1); ++u < i; )
        f[u] = o[u];
      return f[i] = a(c), e(n, this, f);
    };
  }
  return pc = r, pc;
}
var hc, my;
function SP() {
  if (my) return hc;
  my = 1;
  function e(t) {
    return function() {
      return t;
    };
  }
  return hc = e, hc;
}
var vc, gy;
function rx() {
  if (gy) return vc;
  gy = 1;
  var e = gr(), t = function() {
    try {
      var r = e(Object, "defineProperty");
      return r({}, "", {}), r;
    } catch {
    }
  }();
  return vc = t, vc;
}
var yc, by;
function PP() {
  if (by) return yc;
  by = 1;
  var e = SP(), t = rx(), r = cn(), n = t ? function(i, a) {
    return t(i, "toString", {
      configurable: !0,
      enumerable: !1,
      value: e(a),
      writable: !0
    });
  } : r;
  return yc = n, yc;
}
var mc, xy;
function AP() {
  if (xy) return mc;
  xy = 1;
  var e = 800, t = 16, r = Date.now;
  function n(i) {
    var a = 0, o = 0;
    return function() {
      var u = r(), s = t - (u - o);
      if (o = u, s > 0) {
        if (++a >= e)
          return arguments[0];
      } else
        a = 0;
      return i.apply(void 0, arguments);
    };
  }
  return mc = n, mc;
}
var gc, wy;
function EP() {
  if (wy) return gc;
  wy = 1;
  var e = PP(), t = AP(), r = t(e);
  return gc = r, gc;
}
var bc, Oy;
function TP() {
  if (Oy) return bc;
  Oy = 1;
  var e = cn(), t = _P(), r = EP();
  function n(i, a) {
    return r(t(i, a, e), i + "");
  }
  return bc = n, bc;
}
var xc, _y;
function eo() {
  if (_y) return xc;
  _y = 1;
  var e = Qf(), t = bi(), r = pd(), n = Zt();
  function i(a, o, u) {
    if (!n(u))
      return !1;
    var s = typeof o;
    return (s == "number" ? t(u) && r(o, u.length) : s == "string" && o in u) ? e(u[o], a) : !1;
  }
  return xc = i, xc;
}
var wc, Sy;
function jP() {
  if (Sy) return wc;
  Sy = 1;
  var e = Q0(), t = wP(), r = TP(), n = eo(), i = r(function(a, o) {
    if (a == null)
      return [];
    var u = o.length;
    return u > 1 && n(a, o[0], o[1]) ? o = [] : u > 2 && n(o[0], o[1], o[2]) && (o = [o[0]]), t(a, e(o, 1), []);
  });
  return wc = i, wc;
}
var CP = jP();
const md = /* @__PURE__ */ Pe(CP);
function qn(e) {
  "@babel/helpers - typeof";
  return qn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, qn(e);
}
function Fl() {
  return Fl = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Fl.apply(this, arguments);
}
function MP(e, t) {
  return kP(e) || RP(e, t) || $P(e, t) || IP();
}
function IP() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function $P(e, t) {
  if (e) {
    if (typeof e == "string") return Py(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Py(e, t);
  }
}
function Py(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function RP(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function kP(e) {
  if (Array.isArray(e)) return e;
}
function Ay(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Oc(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ay(Object(r), !0).forEach(function(n) {
      NP(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ay(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function NP(e, t, r) {
  return t = DP(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function DP(e) {
  var t = qP(e, "string");
  return qn(t) == "symbol" ? t : t + "";
}
function qP(e, t) {
  if (qn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (qn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function LP(e) {
  return Array.isArray(e) && ke(e[0]) && ke(e[1]) ? e.join(" ~ ") : e;
}
var BP = function(t) {
  var r = t.separator, n = r === void 0 ? " : " : r, i = t.contentStyle, a = i === void 0 ? {} : i, o = t.itemStyle, u = o === void 0 ? {} : o, s = t.labelStyle, c = s === void 0 ? {} : s, f = t.payload, l = t.formatter, d = t.itemSorter, p = t.wrapperClassName, y = t.labelClassName, v = t.label, h = t.labelFormatter, g = t.accessibilityLayer, w = g === void 0 ? !1 : g, b = function() {
    if (f && f.length) {
      var j = {
        padding: 0,
        margin: 0
      }, M = (d ? md(f, d) : f).map(function(R, k) {
        if (R.type === "none")
          return null;
        var q = Oc({
          display: "block",
          paddingTop: 4,
          paddingBottom: 4,
          color: R.color || "#000"
        }, u), L = R.formatter || l || LP, U = R.value, z = R.name, $ = U, D = z;
        if ($ != null && D != null) {
          var B = L(U, z, R, k, f);
          if (Array.isArray(B)) {
            var G = MP(B, 2);
            $ = G[0], D = G[1];
          } else
            $ = B;
        }
        return (
          // eslint-disable-next-line react/no-array-index-key
          /* @__PURE__ */ T.createElement("li", {
            className: "recharts-tooltip-item",
            key: "tooltip-item-".concat(k),
            style: q
          }, ke(D) ? /* @__PURE__ */ T.createElement("span", {
            className: "recharts-tooltip-item-name"
          }, D) : null, ke(D) ? /* @__PURE__ */ T.createElement("span", {
            className: "recharts-tooltip-item-separator"
          }, n) : null, /* @__PURE__ */ T.createElement("span", {
            className: "recharts-tooltip-item-value"
          }, $), /* @__PURE__ */ T.createElement("span", {
            className: "recharts-tooltip-item-unit"
          }, R.unit || ""))
        );
      });
      return /* @__PURE__ */ T.createElement("ul", {
        className: "recharts-tooltip-item-list",
        style: j
      }, M);
    }
    return null;
  }, O = Oc({
    margin: 0,
    padding: 10,
    backgroundColor: "#fff",
    border: "1px solid #ccc",
    whiteSpace: "nowrap"
  }, a), m = Oc({
    margin: 0
  }, c), x = !ce(v), _ = x ? v : "", P = pe("recharts-default-tooltip", p), E = pe("recharts-tooltip-label", y);
  x && h && f !== void 0 && f !== null && (_ = h(v, f));
  var I = w ? {
    role: "status",
    "aria-live": "assertive"
  } : {};
  return /* @__PURE__ */ T.createElement("div", Fl({
    className: P,
    style: O
  }, I), /* @__PURE__ */ T.createElement("p", {
    className: E,
    style: m
  }, /* @__PURE__ */ T.isValidElement(_) ? _ : "".concat(_)), b());
};
function Ln(e) {
  "@babel/helpers - typeof";
  return Ln = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Ln(e);
}
function Mi(e, t, r) {
  return t = FP(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function FP(e) {
  var t = zP(e, "string");
  return Ln(t) == "symbol" ? t : t + "";
}
function zP(e, t) {
  if (Ln(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Ln(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var bn = "recharts-tooltip-wrapper", UP = {
  visibility: "hidden"
};
function WP(e) {
  var t = e.coordinate, r = e.translateX, n = e.translateY;
  return pe(bn, Mi(Mi(Mi(Mi({}, "".concat(bn, "-right"), K(r) && t && K(t.x) && r >= t.x), "".concat(bn, "-left"), K(r) && t && K(t.x) && r < t.x), "".concat(bn, "-bottom"), K(n) && t && K(t.y) && n >= t.y), "".concat(bn, "-top"), K(n) && t && K(t.y) && n < t.y));
}
function Ey(e) {
  var t = e.allowEscapeViewBox, r = e.coordinate, n = e.key, i = e.offsetTopLeft, a = e.position, o = e.reverseDirection, u = e.tooltipDimension, s = e.viewBox, c = e.viewBoxDimension;
  if (a && K(a[n]))
    return a[n];
  var f = r[n] - u - i, l = r[n] + i;
  if (t[n])
    return o[n] ? f : l;
  if (o[n]) {
    var d = f, p = s[n];
    return d < p ? Math.max(l, s[n]) : Math.max(f, s[n]);
  }
  var y = l + u, v = s[n] + c;
  return y > v ? Math.max(f, s[n]) : Math.max(l, s[n]);
}
function GP(e) {
  var t = e.translateX, r = e.translateY, n = e.useTranslate3d;
  return {
    transform: n ? "translate3d(".concat(t, "px, ").concat(r, "px, 0)") : "translate(".concat(t, "px, ").concat(r, "px)")
  };
}
function HP(e) {
  var t = e.allowEscapeViewBox, r = e.coordinate, n = e.offsetTopLeft, i = e.position, a = e.reverseDirection, o = e.tooltipBox, u = e.useTranslate3d, s = e.viewBox, c, f, l;
  return o.height > 0 && o.width > 0 && r ? (f = Ey({
    allowEscapeViewBox: t,
    coordinate: r,
    key: "x",
    offsetTopLeft: n,
    position: i,
    reverseDirection: a,
    tooltipDimension: o.width,
    viewBox: s,
    viewBoxDimension: s.width
  }), l = Ey({
    allowEscapeViewBox: t,
    coordinate: r,
    key: "y",
    offsetTopLeft: n,
    position: i,
    reverseDirection: a,
    tooltipDimension: o.height,
    viewBox: s,
    viewBoxDimension: s.height
  }), c = GP({
    translateX: f,
    translateY: l,
    useTranslate3d: u
  })) : c = UP, {
    cssProperties: c,
    cssClasses: WP({
      translateX: f,
      translateY: l,
      coordinate: r
    })
  };
}
function Lr(e) {
  "@babel/helpers - typeof";
  return Lr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Lr(e);
}
function Ty(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function jy(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ty(Object(r), !0).forEach(function(n) {
      Ul(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ty(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function KP(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function VP(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, ix(n.key), n);
  }
}
function YP(e, t, r) {
  return VP(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function XP(e, t, r) {
  return t = ea(t), ZP(e, nx() ? Reflect.construct(t, r, ea(e).constructor) : t.apply(e, r));
}
function ZP(e, t) {
  if (t && (Lr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return JP(e);
}
function JP(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function nx() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (nx = function() {
    return !!e;
  })();
}
function ea(e) {
  return ea = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, ea(e);
}
function QP(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && zl(e, t);
}
function zl(e, t) {
  return zl = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, zl(e, t);
}
function Ul(e, t, r) {
  return t = ix(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function ix(e) {
  var t = eA(e, "string");
  return Lr(t) == "symbol" ? t : t + "";
}
function eA(e, t) {
  if (Lr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t);
    if (Lr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return String(e);
}
var Cy = 1, tA = /* @__PURE__ */ function(e) {
  function t() {
    var r;
    KP(this, t);
    for (var n = arguments.length, i = new Array(n), a = 0; a < n; a++)
      i[a] = arguments[a];
    return r = XP(this, t, [].concat(i)), Ul(r, "state", {
      dismissed: !1,
      dismissedAtCoordinate: {
        x: 0,
        y: 0
      },
      lastBoundingBox: {
        width: -1,
        height: -1
      }
    }), Ul(r, "handleKeyDown", function(o) {
      if (o.key === "Escape") {
        var u, s, c, f;
        r.setState({
          dismissed: !0,
          dismissedAtCoordinate: {
            x: (u = (s = r.props.coordinate) === null || s === void 0 ? void 0 : s.x) !== null && u !== void 0 ? u : 0,
            y: (c = (f = r.props.coordinate) === null || f === void 0 ? void 0 : f.y) !== null && c !== void 0 ? c : 0
          }
        });
      }
    }), r;
  }
  return QP(t, e), YP(t, [{
    key: "updateBBox",
    value: function() {
      if (this.wrapperNode && this.wrapperNode.getBoundingClientRect) {
        var n = this.wrapperNode.getBoundingClientRect();
        (Math.abs(n.width - this.state.lastBoundingBox.width) > Cy || Math.abs(n.height - this.state.lastBoundingBox.height) > Cy) && this.setState({
          lastBoundingBox: {
            width: n.width,
            height: n.height
          }
        });
      } else (this.state.lastBoundingBox.width !== -1 || this.state.lastBoundingBox.height !== -1) && this.setState({
        lastBoundingBox: {
          width: -1,
          height: -1
        }
      });
    }
  }, {
    key: "componentDidMount",
    value: function() {
      document.addEventListener("keydown", this.handleKeyDown), this.updateBBox();
    }
  }, {
    key: "componentWillUnmount",
    value: function() {
      document.removeEventListener("keydown", this.handleKeyDown);
    }
  }, {
    key: "componentDidUpdate",
    value: function() {
      var n, i;
      this.props.active && this.updateBBox(), this.state.dismissed && (((n = this.props.coordinate) === null || n === void 0 ? void 0 : n.x) !== this.state.dismissedAtCoordinate.x || ((i = this.props.coordinate) === null || i === void 0 ? void 0 : i.y) !== this.state.dismissedAtCoordinate.y) && (this.state.dismissed = !1);
    }
  }, {
    key: "render",
    value: function() {
      var n = this, i = this.props, a = i.active, o = i.allowEscapeViewBox, u = i.animationDuration, s = i.animationEasing, c = i.children, f = i.coordinate, l = i.hasPayload, d = i.isAnimationActive, p = i.offset, y = i.position, v = i.reverseDirection, h = i.useTranslate3d, g = i.viewBox, w = i.wrapperStyle, b = HP({
        allowEscapeViewBox: o,
        coordinate: f,
        offsetTopLeft: p,
        position: y,
        reverseDirection: v,
        tooltipBox: this.state.lastBoundingBox,
        useTranslate3d: h,
        viewBox: g
      }), O = b.cssClasses, m = b.cssProperties, x = jy(jy({
        transition: d && a ? "transform ".concat(u, "ms ").concat(s) : void 0
      }, m), {}, {
        pointerEvents: "none",
        visibility: !this.state.dismissed && a && l ? "visible" : "hidden",
        position: "absolute",
        top: 0,
        left: 0
      }, w);
      return (
        // This element allow listening to the `Escape` key.
        // See https://github.com/recharts/recharts/pull/2925
        /* @__PURE__ */ T.createElement("div", {
          tabIndex: -1,
          className: O,
          style: x,
          ref: function(P) {
            n.wrapperNode = P;
          }
        }, c)
      );
    }
  }]);
}(Xt), rA = function() {
  return !(typeof window < "u" && window.document && window.document.createElement && window.setTimeout);
}, It = {
  isSsr: rA(),
  get: function(t) {
    return It[t];
  },
  set: function(t, r) {
    if (typeof t == "string")
      It[t] = r;
    else {
      var n = Object.keys(t);
      n && n.length && n.forEach(function(i) {
        It[i] = t[i];
      });
    }
  }
};
function Br(e) {
  "@babel/helpers - typeof";
  return Br = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Br(e);
}
function My(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Iy(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? My(Object(r), !0).forEach(function(n) {
      gd(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : My(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function nA(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function iA(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, ox(n.key), n);
  }
}
function aA(e, t, r) {
  return iA(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function oA(e, t, r) {
  return t = ta(t), uA(e, ax() ? Reflect.construct(t, r || [], ta(e).constructor) : t.apply(e, r));
}
function uA(e, t) {
  if (t && (Br(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return sA(e);
}
function sA(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function ax() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (ax = function() {
    return !!e;
  })();
}
function ta(e) {
  return ta = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, ta(e);
}
function cA(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Wl(e, t);
}
function Wl(e, t) {
  return Wl = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Wl(e, t);
}
function gd(e, t, r) {
  return t = ox(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function ox(e) {
  var t = lA(e, "string");
  return Br(t) == "symbol" ? t : t + "";
}
function lA(e, t) {
  if (Br(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Br(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function fA(e) {
  return e.dataKey;
}
function dA(e, t) {
  return /* @__PURE__ */ T.isValidElement(e) ? /* @__PURE__ */ T.cloneElement(e, t) : typeof e == "function" ? /* @__PURE__ */ T.createElement(e, t) : /* @__PURE__ */ T.createElement(BP, t);
}
var gt = /* @__PURE__ */ function(e) {
  function t() {
    return nA(this, t), oA(this, t, arguments);
  }
  return cA(t, e), aA(t, [{
    key: "render",
    value: function() {
      var n = this, i = this.props, a = i.active, o = i.allowEscapeViewBox, u = i.animationDuration, s = i.animationEasing, c = i.content, f = i.coordinate, l = i.filterNull, d = i.isAnimationActive, p = i.offset, y = i.payload, v = i.payloadUniqBy, h = i.position, g = i.reverseDirection, w = i.useTranslate3d, b = i.viewBox, O = i.wrapperStyle, m = y ?? [];
      l && m.length && (m = X0(y.filter(function(_) {
        return _.value != null && (_.hide !== !0 || n.props.includeHidden);
      }), v, fA));
      var x = m.length > 0;
      return /* @__PURE__ */ T.createElement(tA, {
        allowEscapeViewBox: o,
        animationDuration: u,
        animationEasing: s,
        isAnimationActive: d,
        active: a,
        coordinate: f,
        hasPayload: x,
        offset: p,
        position: h,
        reverseDirection: g,
        useTranslate3d: w,
        viewBox: b,
        wrapperStyle: O
      }, dA(c, Iy(Iy({}, this.props), {}, {
        payload: m
      })));
    }
  }]);
}(Xt);
gd(gt, "displayName", "Tooltip");
gd(gt, "defaultProps", {
  accessibilityLayer: !1,
  allowEscapeViewBox: {
    x: !1,
    y: !1
  },
  animationDuration: 400,
  animationEasing: "ease",
  contentStyle: {},
  coordinate: {
    x: 0,
    y: 0
  },
  cursor: !0,
  cursorStyle: {},
  filterNull: !0,
  isAnimationActive: !It.isSsr,
  itemStyle: {},
  labelStyle: {},
  offset: 10,
  reverseDirection: {
    x: !1,
    y: !1
  },
  separator: " : ",
  trigger: "hover",
  useTranslate3d: !1,
  viewBox: {
    x: 0,
    y: 0,
    height: 0,
    width: 0
  },
  wrapperStyle: {}
});
var _c, $y;
function pA() {
  if ($y) return _c;
  $y = 1;
  var e = St(), t = function() {
    return e.Date.now();
  };
  return _c = t, _c;
}
var Sc, Ry;
function hA() {
  if (Ry) return Sc;
  Ry = 1;
  var e = /\s/;
  function t(r) {
    for (var n = r.length; n-- && e.test(r.charAt(n)); )
      ;
    return n;
  }
  return Sc = t, Sc;
}
var Pc, ky;
function vA() {
  if (ky) return Pc;
  ky = 1;
  var e = hA(), t = /^\s+/;
  function r(n) {
    return n && n.slice(0, e(n) + 1).replace(t, "");
  }
  return Pc = r, Pc;
}
var Ac, Ny;
function ux() {
  if (Ny) return Ac;
  Ny = 1;
  var e = vA(), t = Zt(), r = un(), n = NaN, i = /^[-+]0x[0-9a-f]+$/i, a = /^0b[01]+$/i, o = /^0o[0-7]+$/i, u = parseInt;
  function s(c) {
    if (typeof c == "number")
      return c;
    if (r(c))
      return n;
    if (t(c)) {
      var f = typeof c.valueOf == "function" ? c.valueOf() : c;
      c = t(f) ? f + "" : f;
    }
    if (typeof c != "string")
      return c === 0 ? c : +c;
    c = e(c);
    var l = a.test(c);
    return l || o.test(c) ? u(c.slice(2), l ? 2 : 8) : i.test(c) ? n : +c;
  }
  return Ac = s, Ac;
}
var Ec, Dy;
function yA() {
  if (Dy) return Ec;
  Dy = 1;
  var e = Zt(), t = pA(), r = ux(), n = "Expected a function", i = Math.max, a = Math.min;
  function o(u, s, c) {
    var f, l, d, p, y, v, h = 0, g = !1, w = !1, b = !0;
    if (typeof u != "function")
      throw new TypeError(n);
    s = r(s) || 0, e(c) && (g = !!c.leading, w = "maxWait" in c, d = w ? i(r(c.maxWait) || 0, s) : d, b = "trailing" in c ? !!c.trailing : b);
    function O(M) {
      var R = f, k = l;
      return f = l = void 0, h = M, p = u.apply(k, R), p;
    }
    function m(M) {
      return h = M, y = setTimeout(P, s), g ? O(M) : p;
    }
    function x(M) {
      var R = M - v, k = M - h, q = s - R;
      return w ? a(q, d - k) : q;
    }
    function _(M) {
      var R = M - v, k = M - h;
      return v === void 0 || R >= s || R < 0 || w && k >= d;
    }
    function P() {
      var M = t();
      if (_(M))
        return E(M);
      y = setTimeout(P, x(M));
    }
    function E(M) {
      return y = void 0, b && f ? O(M) : (f = l = void 0, p);
    }
    function I() {
      y !== void 0 && clearTimeout(y), h = 0, f = v = l = y = void 0;
    }
    function S() {
      return y === void 0 ? p : E(t());
    }
    function j() {
      var M = t(), R = _(M);
      if (f = arguments, l = this, v = M, R) {
        if (y === void 0)
          return m(v);
        if (w)
          return clearTimeout(y), y = setTimeout(P, s), O(v);
      }
      return y === void 0 && (y = setTimeout(P, s)), p;
    }
    return j.cancel = I, j.flush = S, j;
  }
  return Ec = o, Ec;
}
var Tc, qy;
function mA() {
  if (qy) return Tc;
  qy = 1;
  var e = yA(), t = Zt(), r = "Expected a function";
  function n(i, a, o) {
    var u = !0, s = !0;
    if (typeof i != "function")
      throw new TypeError(r);
    return t(o) && (u = "leading" in o ? !!o.leading : u, s = "trailing" in o ? !!o.trailing : s), e(i, a, {
      leading: u,
      maxWait: a,
      trailing: s
    });
  }
  return Tc = n, Tc;
}
var gA = mA();
const bA = /* @__PURE__ */ Pe(gA);
var sx = function(t) {
  return null;
};
sx.displayName = "Cell";
function Bn(e) {
  "@babel/helpers - typeof";
  return Bn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Bn(e);
}
function Ly(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Gl(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ly(Object(r), !0).forEach(function(n) {
      xA(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ly(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function xA(e, t, r) {
  return t = wA(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function wA(e) {
  var t = OA(e, "string");
  return Bn(t) == "symbol" ? t : t + "";
}
function OA(e, t) {
  if (Bn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Bn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var _r = {
  widthCache: {},
  cacheCount: 0
}, _A = 2e3, SA = {
  position: "absolute",
  top: "-20000px",
  left: 0,
  padding: 0,
  margin: 0,
  border: "none",
  whiteSpace: "pre"
}, By = "recharts_measurement_span";
function PA(e) {
  var t = Gl({}, e);
  return Object.keys(t).forEach(function(r) {
    t[r] || delete t[r];
  }), t;
}
var Mn = function(t) {
  var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
  if (t == null || It.isSsr)
    return {
      width: 0,
      height: 0
    };
  var n = PA(r), i = JSON.stringify({
    text: t,
    copyStyle: n
  });
  if (_r.widthCache[i])
    return _r.widthCache[i];
  try {
    var a = document.getElementById(By);
    a || (a = document.createElement("span"), a.setAttribute("id", By), a.setAttribute("aria-hidden", "true"), document.body.appendChild(a));
    var o = Gl(Gl({}, SA), n);
    Object.assign(a.style, o), a.textContent = "".concat(t);
    var u = a.getBoundingClientRect(), s = {
      width: u.width,
      height: u.height
    };
    return _r.widthCache[i] = s, ++_r.cacheCount > _A && (_r.cacheCount = 0, _r.widthCache = {}), s;
  } catch {
    return {
      width: 0,
      height: 0
    };
  }
}, AA = function(t) {
  return {
    top: t.top + window.scrollY - document.documentElement.clientTop,
    left: t.left + window.scrollX - document.documentElement.clientLeft
  };
};
function Fn(e) {
  "@babel/helpers - typeof";
  return Fn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Fn(e);
}
function ra(e, t) {
  return CA(e) || jA(e, t) || TA(e, t) || EA();
}
function EA() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function TA(e, t) {
  if (e) {
    if (typeof e == "string") return Fy(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Fy(e, t);
  }
}
function Fy(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function jA(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t === 0) {
        if (Object(r) !== r) return;
        s = !1;
      } else for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function CA(e) {
  if (Array.isArray(e)) return e;
}
function MA(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function zy(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, $A(n.key), n);
  }
}
function IA(e, t, r) {
  return zy(e.prototype, t), zy(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function $A(e) {
  var t = RA(e, "string");
  return Fn(t) == "symbol" ? t : t + "";
}
function RA(e, t) {
  if (Fn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t);
    if (Fn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return String(e);
}
var Uy = /(-?\d+(?:\.\d+)?[a-zA-Z%]*)([*/])(-?\d+(?:\.\d+)?[a-zA-Z%]*)/, Wy = /(-?\d+(?:\.\d+)?[a-zA-Z%]*)([+-])(-?\d+(?:\.\d+)?[a-zA-Z%]*)/, kA = /^px|cm|vh|vw|em|rem|%|mm|in|pt|pc|ex|ch|vmin|vmax|Q$/, NA = /(-?\d+(?:\.\d+)?)([a-zA-Z%]+)?/, cx = {
  cm: 96 / 2.54,
  mm: 96 / 25.4,
  pt: 96 / 72,
  pc: 96 / 6,
  in: 96,
  Q: 96 / (2.54 * 40),
  px: 1
}, DA = Object.keys(cx), Tr = "NaN";
function qA(e, t) {
  return e * cx[t];
}
var Ii = /* @__PURE__ */ function() {
  function e(t, r) {
    MA(this, e), this.num = t, this.unit = r, this.num = t, this.unit = r, Number.isNaN(t) && (this.unit = ""), r !== "" && !kA.test(r) && (this.num = NaN, this.unit = ""), DA.includes(r) && (this.num = qA(t, r), this.unit = "px");
  }
  return IA(e, [{
    key: "add",
    value: function(r) {
      return this.unit !== r.unit ? new e(NaN, "") : new e(this.num + r.num, this.unit);
    }
  }, {
    key: "subtract",
    value: function(r) {
      return this.unit !== r.unit ? new e(NaN, "") : new e(this.num - r.num, this.unit);
    }
  }, {
    key: "multiply",
    value: function(r) {
      return this.unit !== "" && r.unit !== "" && this.unit !== r.unit ? new e(NaN, "") : new e(this.num * r.num, this.unit || r.unit);
    }
  }, {
    key: "divide",
    value: function(r) {
      return this.unit !== "" && r.unit !== "" && this.unit !== r.unit ? new e(NaN, "") : new e(this.num / r.num, this.unit || r.unit);
    }
  }, {
    key: "toString",
    value: function() {
      return "".concat(this.num).concat(this.unit);
    }
  }, {
    key: "isNaN",
    value: function() {
      return Number.isNaN(this.num);
    }
  }], [{
    key: "parse",
    value: function(r) {
      var n, i = (n = NA.exec(r)) !== null && n !== void 0 ? n : [], a = ra(i, 3), o = a[1], u = a[2];
      return new e(parseFloat(o), u ?? "");
    }
  }]);
}();
function lx(e) {
  if (e.includes(Tr))
    return Tr;
  for (var t = e; t.includes("*") || t.includes("/"); ) {
    var r, n = (r = Uy.exec(t)) !== null && r !== void 0 ? r : [], i = ra(n, 4), a = i[1], o = i[2], u = i[3], s = Ii.parse(a ?? ""), c = Ii.parse(u ?? ""), f = o === "*" ? s.multiply(c) : s.divide(c);
    if (f.isNaN())
      return Tr;
    t = t.replace(Uy, f.toString());
  }
  for (; t.includes("+") || /.-\d+(?:\.\d+)?/.test(t); ) {
    var l, d = (l = Wy.exec(t)) !== null && l !== void 0 ? l : [], p = ra(d, 4), y = p[1], v = p[2], h = p[3], g = Ii.parse(y ?? ""), w = Ii.parse(h ?? ""), b = v === "+" ? g.add(w) : g.subtract(w);
    if (b.isNaN())
      return Tr;
    t = t.replace(Wy, b.toString());
  }
  return t;
}
var Gy = /\(([^()]*)\)/;
function LA(e) {
  for (var t = e; t.includes("("); ) {
    var r = Gy.exec(t), n = ra(r, 2), i = n[1];
    t = t.replace(Gy, lx(i));
  }
  return t;
}
function BA(e) {
  var t = e.replace(/\s+/g, "");
  return t = LA(t), t = lx(t), t;
}
function FA(e) {
  try {
    return BA(e);
  } catch {
    return Tr;
  }
}
function jc(e) {
  var t = FA(e.slice(5, -1));
  return t === Tr ? "" : t;
}
var zA = ["x", "y", "lineHeight", "capHeight", "scaleToFit", "textAnchor", "verticalAnchor", "fill"], UA = ["dx", "dy", "angle", "className", "breakAll"];
function Hl() {
  return Hl = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Hl.apply(this, arguments);
}
function Hy(e, t) {
  if (e == null) return {};
  var r = WA(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function WA(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function Ky(e, t) {
  return VA(e) || KA(e, t) || HA(e, t) || GA();
}
function GA() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function HA(e, t) {
  if (e) {
    if (typeof e == "string") return Vy(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Vy(e, t);
  }
}
function Vy(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function KA(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t === 0) {
        if (Object(r) !== r) return;
        s = !1;
      } else for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function VA(e) {
  if (Array.isArray(e)) return e;
}
var fx = /[ \f\n\r\t\v\u2028\u2029]+/, dx = function(t) {
  var r = t.children, n = t.breakAll, i = t.style;
  try {
    var a = [];
    ce(r) || (n ? a = r.toString().split("") : a = r.toString().split(fx));
    var o = a.map(function(s) {
      return {
        word: s,
        width: Mn(s, i).width
      };
    }), u = n ? 0 : Mn(" ", i).width;
    return {
      wordsWithComputedWidth: o,
      spaceWidth: u
    };
  } catch {
    return null;
  }
}, YA = function(t, r, n, i, a) {
  var o = t.maxLines, u = t.children, s = t.style, c = t.breakAll, f = K(o), l = u, d = function() {
    var k = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : [];
    return k.reduce(function(q, L) {
      var U = L.word, z = L.width, $ = q[q.length - 1];
      if ($ && (i == null || a || $.width + z + n < Number(i)))
        $.words.push(U), $.width += z + n;
      else {
        var D = {
          words: [U],
          width: z
        };
        q.push(D);
      }
      return q;
    }, []);
  }, p = d(r), y = function(k) {
    return k.reduce(function(q, L) {
      return q.width > L.width ? q : L;
    });
  };
  if (!f)
    return p;
  for (var v = "…", h = function(k) {
    var q = l.slice(0, k), L = dx({
      breakAll: c,
      style: s,
      children: q + v
    }).wordsWithComputedWidth, U = d(L), z = U.length > o || y(U).width > Number(i);
    return [z, U];
  }, g = 0, w = l.length - 1, b = 0, O; g <= w && b <= l.length - 1; ) {
    var m = Math.floor((g + w) / 2), x = m - 1, _ = h(x), P = Ky(_, 2), E = P[0], I = P[1], S = h(m), j = Ky(S, 1), M = j[0];
    if (!E && !M && (g = m + 1), E && M && (w = m - 1), !E && M) {
      O = I;
      break;
    }
    b++;
  }
  return O || p;
}, Yy = function(t) {
  var r = ce(t) ? [] : t.toString().split(fx);
  return [{
    words: r
  }];
}, XA = function(t) {
  var r = t.width, n = t.scaleToFit, i = t.children, a = t.style, o = t.breakAll, u = t.maxLines;
  if ((r || n) && !It.isSsr) {
    var s, c, f = dx({
      breakAll: o,
      children: i,
      style: a
    });
    if (f) {
      var l = f.wordsWithComputedWidth, d = f.spaceWidth;
      s = l, c = d;
    } else
      return Yy(i);
    return YA({
      breakAll: o,
      children: i,
      maxLines: u,
      style: a
    }, s, c, r, n);
  }
  return Yy(i);
}, Xy = "#808080", na = function(t) {
  var r = t.x, n = r === void 0 ? 0 : r, i = t.y, a = i === void 0 ? 0 : i, o = t.lineHeight, u = o === void 0 ? "1em" : o, s = t.capHeight, c = s === void 0 ? "0.71em" : s, f = t.scaleToFit, l = f === void 0 ? !1 : f, d = t.textAnchor, p = d === void 0 ? "start" : d, y = t.verticalAnchor, v = y === void 0 ? "end" : y, h = t.fill, g = h === void 0 ? Xy : h, w = Hy(t, zA), b = FO(function() {
    return XA({
      breakAll: w.breakAll,
      children: w.children,
      maxLines: w.maxLines,
      scaleToFit: l,
      style: w.style,
      width: w.width
    });
  }, [w.breakAll, w.children, w.maxLines, l, w.style, w.width]), O = w.dx, m = w.dy, x = w.angle, _ = w.className, P = w.breakAll, E = Hy(w, UA);
  if (!ke(n) || !ke(a))
    return null;
  var I = n + (K(O) ? O : 0), S = a + (K(m) ? m : 0), j;
  switch (v) {
    case "start":
      j = jc("calc(".concat(c, ")"));
      break;
    case "middle":
      j = jc("calc(".concat((b.length - 1) / 2, " * -").concat(u, " + (").concat(c, " / 2))"));
      break;
    default:
      j = jc("calc(".concat(b.length - 1, " * -").concat(u, ")"));
      break;
  }
  var M = [];
  if (l) {
    var R = b[0].width, k = w.width;
    M.push("scale(".concat((K(k) ? k / R : 1) / R, ")"));
  }
  return x && M.push("rotate(".concat(x, ", ").concat(I, ", ").concat(S, ")")), M.length && (E.transform = M.join(" ")), /* @__PURE__ */ T.createElement("text", Hl({}, fe(E, !0), {
    x: I,
    y: S,
    className: pe("recharts-text", _),
    textAnchor: p,
    fill: g.includes("url") ? Xy : g
  }), b.map(function(q, L) {
    var U = q.words.join(P ? "" : " ");
    return (
      // duplicate words will cause duplicate keys
      // eslint-disable-next-line react/no-array-index-key
      /* @__PURE__ */ T.createElement("tspan", {
        x: I,
        dy: L === 0 ? j : u,
        key: "".concat(U, "-").concat(L)
      }, U)
    );
  }));
};
function Kt(e, t) {
  return e == null || t == null ? NaN : e < t ? -1 : e > t ? 1 : e >= t ? 0 : NaN;
}
function ZA(e, t) {
  return e == null || t == null ? NaN : t < e ? -1 : t > e ? 1 : t >= e ? 0 : NaN;
}
function bd(e) {
  let t, r, n;
  e.length !== 2 ? (t = Kt, r = (u, s) => Kt(e(u), s), n = (u, s) => e(u) - s) : (t = e === Kt || e === ZA ? e : JA, r = e, n = e);
  function i(u, s, c = 0, f = u.length) {
    if (c < f) {
      if (t(s, s) !== 0) return f;
      do {
        const l = c + f >>> 1;
        r(u[l], s) < 0 ? c = l + 1 : f = l;
      } while (c < f);
    }
    return c;
  }
  function a(u, s, c = 0, f = u.length) {
    if (c < f) {
      if (t(s, s) !== 0) return f;
      do {
        const l = c + f >>> 1;
        r(u[l], s) <= 0 ? c = l + 1 : f = l;
      } while (c < f);
    }
    return c;
  }
  function o(u, s, c = 0, f = u.length) {
    const l = i(u, s, c, f - 1);
    return l > c && n(u[l - 1], s) > -n(u[l], s) ? l - 1 : l;
  }
  return { left: i, center: o, right: a };
}
function JA() {
  return 0;
}
function px(e) {
  return e === null ? NaN : +e;
}
function* QA(e, t) {
  for (let r of e)
    r != null && (r = +r) >= r && (yield r);
}
const eE = bd(Kt), xi = eE.right;
bd(px).center;
class Zy extends Map {
  constructor(t, r = nE) {
    if (super(), Object.defineProperties(this, { _intern: { value: /* @__PURE__ */ new Map() }, _key: { value: r } }), t != null) for (const [n, i] of t) this.set(n, i);
  }
  get(t) {
    return super.get(Jy(this, t));
  }
  has(t) {
    return super.has(Jy(this, t));
  }
  set(t, r) {
    return super.set(tE(this, t), r);
  }
  delete(t) {
    return super.delete(rE(this, t));
  }
}
function Jy({ _intern: e, _key: t }, r) {
  const n = t(r);
  return e.has(n) ? e.get(n) : r;
}
function tE({ _intern: e, _key: t }, r) {
  const n = t(r);
  return e.has(n) ? e.get(n) : (e.set(n, r), r);
}
function rE({ _intern: e, _key: t }, r) {
  const n = t(r);
  return e.has(n) && (r = e.get(n), e.delete(n)), r;
}
function nE(e) {
  return e !== null && typeof e == "object" ? e.valueOf() : e;
}
function iE(e = Kt) {
  if (e === Kt) return hx;
  if (typeof e != "function") throw new TypeError("compare is not a function");
  return (t, r) => {
    const n = e(t, r);
    return n || n === 0 ? n : (e(r, r) === 0) - (e(t, t) === 0);
  };
}
function hx(e, t) {
  return (e == null || !(e >= e)) - (t == null || !(t >= t)) || (e < t ? -1 : e > t ? 1 : 0);
}
const aE = Math.sqrt(50), oE = Math.sqrt(10), uE = Math.sqrt(2);
function ia(e, t, r) {
  const n = (t - e) / Math.max(0, r), i = Math.floor(Math.log10(n)), a = n / Math.pow(10, i), o = a >= aE ? 10 : a >= oE ? 5 : a >= uE ? 2 : 1;
  let u, s, c;
  return i < 0 ? (c = Math.pow(10, -i) / o, u = Math.round(e * c), s = Math.round(t * c), u / c < e && ++u, s / c > t && --s, c = -c) : (c = Math.pow(10, i) * o, u = Math.round(e / c), s = Math.round(t / c), u * c < e && ++u, s * c > t && --s), s < u && 0.5 <= r && r < 2 ? ia(e, t, r * 2) : [u, s, c];
}
function Kl(e, t, r) {
  if (t = +t, e = +e, r = +r, !(r > 0)) return [];
  if (e === t) return [e];
  const n = t < e, [i, a, o] = n ? ia(t, e, r) : ia(e, t, r);
  if (!(a >= i)) return [];
  const u = a - i + 1, s = new Array(u);
  if (n)
    if (o < 0) for (let c = 0; c < u; ++c) s[c] = (a - c) / -o;
    else for (let c = 0; c < u; ++c) s[c] = (a - c) * o;
  else if (o < 0) for (let c = 0; c < u; ++c) s[c] = (i + c) / -o;
  else for (let c = 0; c < u; ++c) s[c] = (i + c) * o;
  return s;
}
function Vl(e, t, r) {
  return t = +t, e = +e, r = +r, ia(e, t, r)[2];
}
function Yl(e, t, r) {
  t = +t, e = +e, r = +r;
  const n = t < e, i = n ? Vl(t, e, r) : Vl(e, t, r);
  return (n ? -1 : 1) * (i < 0 ? 1 / -i : i);
}
function Qy(e, t) {
  let r;
  for (const n of e)
    n != null && (r < n || r === void 0 && n >= n) && (r = n);
  return r;
}
function em(e, t) {
  let r;
  for (const n of e)
    n != null && (r > n || r === void 0 && n >= n) && (r = n);
  return r;
}
function vx(e, t, r = 0, n = 1 / 0, i) {
  if (t = Math.floor(t), r = Math.floor(Math.max(0, r)), n = Math.floor(Math.min(e.length - 1, n)), !(r <= t && t <= n)) return e;
  for (i = i === void 0 ? hx : iE(i); n > r; ) {
    if (n - r > 600) {
      const s = n - r + 1, c = t - r + 1, f = Math.log(s), l = 0.5 * Math.exp(2 * f / 3), d = 0.5 * Math.sqrt(f * l * (s - l) / s) * (c - s / 2 < 0 ? -1 : 1), p = Math.max(r, Math.floor(t - c * l / s + d)), y = Math.min(n, Math.floor(t + (s - c) * l / s + d));
      vx(e, t, p, y, i);
    }
    const a = e[t];
    let o = r, u = n;
    for (xn(e, r, t), i(e[n], a) > 0 && xn(e, r, n); o < u; ) {
      for (xn(e, o, u), ++o, --u; i(e[o], a) < 0; ) ++o;
      for (; i(e[u], a) > 0; ) --u;
    }
    i(e[r], a) === 0 ? xn(e, r, u) : (++u, xn(e, u, n)), u <= t && (r = u + 1), t <= u && (n = u - 1);
  }
  return e;
}
function xn(e, t, r) {
  const n = e[t];
  e[t] = e[r], e[r] = n;
}
function sE(e, t, r) {
  if (e = Float64Array.from(QA(e)), !(!(n = e.length) || isNaN(t = +t))) {
    if (t <= 0 || n < 2) return em(e);
    if (t >= 1) return Qy(e);
    var n, i = (n - 1) * t, a = Math.floor(i), o = Qy(vx(e, a).subarray(0, a + 1)), u = em(e.subarray(a + 1));
    return o + (u - o) * (i - a);
  }
}
function cE(e, t, r = px) {
  if (!(!(n = e.length) || isNaN(t = +t))) {
    if (t <= 0 || n < 2) return +r(e[0], 0, e);
    if (t >= 1) return +r(e[n - 1], n - 1, e);
    var n, i = (n - 1) * t, a = Math.floor(i), o = +r(e[a], a, e), u = +r(e[a + 1], a + 1, e);
    return o + (u - o) * (i - a);
  }
}
function lE(e, t, r) {
  e = +e, t = +t, r = (i = arguments.length) < 2 ? (t = e, e = 0, 1) : i < 3 ? 1 : +r;
  for (var n = -1, i = Math.max(0, Math.ceil((t - e) / r)) | 0, a = new Array(i); ++n < i; )
    a[n] = e + n * r;
  return a;
}
function st(e, t) {
  switch (arguments.length) {
    case 0:
      break;
    case 1:
      this.range(e);
      break;
    default:
      this.range(t).domain(e);
      break;
  }
  return this;
}
function Ft(e, t) {
  switch (arguments.length) {
    case 0:
      break;
    case 1: {
      typeof e == "function" ? this.interpolator(e) : this.range(e);
      break;
    }
    default: {
      this.domain(e), typeof t == "function" ? this.interpolator(t) : this.range(t);
      break;
    }
  }
  return this;
}
const Xl = Symbol("implicit");
function xd() {
  var e = new Zy(), t = [], r = [], n = Xl;
  function i(a) {
    let o = e.get(a);
    if (o === void 0) {
      if (n !== Xl) return n;
      e.set(a, o = t.push(a) - 1);
    }
    return r[o % r.length];
  }
  return i.domain = function(a) {
    if (!arguments.length) return t.slice();
    t = [], e = new Zy();
    for (const o of a)
      e.has(o) || e.set(o, t.push(o) - 1);
    return i;
  }, i.range = function(a) {
    return arguments.length ? (r = Array.from(a), i) : r.slice();
  }, i.unknown = function(a) {
    return arguments.length ? (n = a, i) : n;
  }, i.copy = function() {
    return xd(t, r).unknown(n);
  }, st.apply(i, arguments), i;
}
function zn() {
  var e = xd().unknown(void 0), t = e.domain, r = e.range, n = 0, i = 1, a, o, u = !1, s = 0, c = 0, f = 0.5;
  delete e.unknown;
  function l() {
    var d = t().length, p = i < n, y = p ? i : n, v = p ? n : i;
    a = (v - y) / Math.max(1, d - s + c * 2), u && (a = Math.floor(a)), y += (v - y - a * (d - s)) * f, o = a * (1 - s), u && (y = Math.round(y), o = Math.round(o));
    var h = lE(d).map(function(g) {
      return y + a * g;
    });
    return r(p ? h.reverse() : h);
  }
  return e.domain = function(d) {
    return arguments.length ? (t(d), l()) : t();
  }, e.range = function(d) {
    return arguments.length ? ([n, i] = d, n = +n, i = +i, l()) : [n, i];
  }, e.rangeRound = function(d) {
    return [n, i] = d, n = +n, i = +i, u = !0, l();
  }, e.bandwidth = function() {
    return o;
  }, e.step = function() {
    return a;
  }, e.round = function(d) {
    return arguments.length ? (u = !!d, l()) : u;
  }, e.padding = function(d) {
    return arguments.length ? (s = Math.min(1, c = +d), l()) : s;
  }, e.paddingInner = function(d) {
    return arguments.length ? (s = Math.min(1, d), l()) : s;
  }, e.paddingOuter = function(d) {
    return arguments.length ? (c = +d, l()) : c;
  }, e.align = function(d) {
    return arguments.length ? (f = Math.max(0, Math.min(1, d)), l()) : f;
  }, e.copy = function() {
    return zn(t(), [n, i]).round(u).paddingInner(s).paddingOuter(c).align(f);
  }, st.apply(l(), arguments);
}
function yx(e) {
  var t = e.copy;
  return e.padding = e.paddingOuter, delete e.paddingInner, delete e.paddingOuter, e.copy = function() {
    return yx(t());
  }, e;
}
function In() {
  return yx(zn.apply(null, arguments).paddingInner(1));
}
function wd(e, t, r) {
  e.prototype = t.prototype = r, r.constructor = e;
}
function mx(e, t) {
  var r = Object.create(e.prototype);
  for (var n in t) r[n] = t[n];
  return r;
}
function wi() {
}
var Un = 0.7, aa = 1 / Un, $r = "\\s*([+-]?\\d+)\\s*", Wn = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)\\s*", wt = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)%\\s*", fE = /^#([0-9a-f]{3,8})$/, dE = new RegExp(`^rgb\\(${$r},${$r},${$r}\\)$`), pE = new RegExp(`^rgb\\(${wt},${wt},${wt}\\)$`), hE = new RegExp(`^rgba\\(${$r},${$r},${$r},${Wn}\\)$`), vE = new RegExp(`^rgba\\(${wt},${wt},${wt},${Wn}\\)$`), yE = new RegExp(`^hsl\\(${Wn},${wt},${wt}\\)$`), mE = new RegExp(`^hsla\\(${Wn},${wt},${wt},${Wn}\\)$`), tm = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
};
wd(wi, Gn, {
  copy(e) {
    return Object.assign(new this.constructor(), this, e);
  },
  displayable() {
    return this.rgb().displayable();
  },
  hex: rm,
  // Deprecated! Use color.formatHex.
  formatHex: rm,
  formatHex8: gE,
  formatHsl: bE,
  formatRgb: nm,
  toString: nm
});
function rm() {
  return this.rgb().formatHex();
}
function gE() {
  return this.rgb().formatHex8();
}
function bE() {
  return gx(this).formatHsl();
}
function nm() {
  return this.rgb().formatRgb();
}
function Gn(e) {
  var t, r;
  return e = (e + "").trim().toLowerCase(), (t = fE.exec(e)) ? (r = t[1].length, t = parseInt(t[1], 16), r === 6 ? im(t) : r === 3 ? new Ve(t >> 8 & 15 | t >> 4 & 240, t >> 4 & 15 | t & 240, (t & 15) << 4 | t & 15, 1) : r === 8 ? $i(t >> 24 & 255, t >> 16 & 255, t >> 8 & 255, (t & 255) / 255) : r === 4 ? $i(t >> 12 & 15 | t >> 8 & 240, t >> 8 & 15 | t >> 4 & 240, t >> 4 & 15 | t & 240, ((t & 15) << 4 | t & 15) / 255) : null) : (t = dE.exec(e)) ? new Ve(t[1], t[2], t[3], 1) : (t = pE.exec(e)) ? new Ve(t[1] * 255 / 100, t[2] * 255 / 100, t[3] * 255 / 100, 1) : (t = hE.exec(e)) ? $i(t[1], t[2], t[3], t[4]) : (t = vE.exec(e)) ? $i(t[1] * 255 / 100, t[2] * 255 / 100, t[3] * 255 / 100, t[4]) : (t = yE.exec(e)) ? um(t[1], t[2] / 100, t[3] / 100, 1) : (t = mE.exec(e)) ? um(t[1], t[2] / 100, t[3] / 100, t[4]) : tm.hasOwnProperty(e) ? im(tm[e]) : e === "transparent" ? new Ve(NaN, NaN, NaN, 0) : null;
}
function im(e) {
  return new Ve(e >> 16 & 255, e >> 8 & 255, e & 255, 1);
}
function $i(e, t, r, n) {
  return n <= 0 && (e = t = r = NaN), new Ve(e, t, r, n);
}
function xE(e) {
  return e instanceof wi || (e = Gn(e)), e ? (e = e.rgb(), new Ve(e.r, e.g, e.b, e.opacity)) : new Ve();
}
function Zl(e, t, r, n) {
  return arguments.length === 1 ? xE(e) : new Ve(e, t, r, n ?? 1);
}
function Ve(e, t, r, n) {
  this.r = +e, this.g = +t, this.b = +r, this.opacity = +n;
}
wd(Ve, Zl, mx(wi, {
  brighter(e) {
    return e = e == null ? aa : Math.pow(aa, e), new Ve(this.r * e, this.g * e, this.b * e, this.opacity);
  },
  darker(e) {
    return e = e == null ? Un : Math.pow(Un, e), new Ve(this.r * e, this.g * e, this.b * e, this.opacity);
  },
  rgb() {
    return this;
  },
  clamp() {
    return new Ve(dr(this.r), dr(this.g), dr(this.b), oa(this.opacity));
  },
  displayable() {
    return -0.5 <= this.r && this.r < 255.5 && -0.5 <= this.g && this.g < 255.5 && -0.5 <= this.b && this.b < 255.5 && 0 <= this.opacity && this.opacity <= 1;
  },
  hex: am,
  // Deprecated! Use color.formatHex.
  formatHex: am,
  formatHex8: wE,
  formatRgb: om,
  toString: om
}));
function am() {
  return `#${ur(this.r)}${ur(this.g)}${ur(this.b)}`;
}
function wE() {
  return `#${ur(this.r)}${ur(this.g)}${ur(this.b)}${ur((isNaN(this.opacity) ? 1 : this.opacity) * 255)}`;
}
function om() {
  const e = oa(this.opacity);
  return `${e === 1 ? "rgb(" : "rgba("}${dr(this.r)}, ${dr(this.g)}, ${dr(this.b)}${e === 1 ? ")" : `, ${e})`}`;
}
function oa(e) {
  return isNaN(e) ? 1 : Math.max(0, Math.min(1, e));
}
function dr(e) {
  return Math.max(0, Math.min(255, Math.round(e) || 0));
}
function ur(e) {
  return e = dr(e), (e < 16 ? "0" : "") + e.toString(16);
}
function um(e, t, r, n) {
  return n <= 0 ? e = t = r = NaN : r <= 0 || r >= 1 ? e = t = NaN : t <= 0 && (e = NaN), new vt(e, t, r, n);
}
function gx(e) {
  if (e instanceof vt) return new vt(e.h, e.s, e.l, e.opacity);
  if (e instanceof wi || (e = Gn(e)), !e) return new vt();
  if (e instanceof vt) return e;
  e = e.rgb();
  var t = e.r / 255, r = e.g / 255, n = e.b / 255, i = Math.min(t, r, n), a = Math.max(t, r, n), o = NaN, u = a - i, s = (a + i) / 2;
  return u ? (t === a ? o = (r - n) / u + (r < n) * 6 : r === a ? o = (n - t) / u + 2 : o = (t - r) / u + 4, u /= s < 0.5 ? a + i : 2 - a - i, o *= 60) : u = s > 0 && s < 1 ? 0 : o, new vt(o, u, s, e.opacity);
}
function OE(e, t, r, n) {
  return arguments.length === 1 ? gx(e) : new vt(e, t, r, n ?? 1);
}
function vt(e, t, r, n) {
  this.h = +e, this.s = +t, this.l = +r, this.opacity = +n;
}
wd(vt, OE, mx(wi, {
  brighter(e) {
    return e = e == null ? aa : Math.pow(aa, e), new vt(this.h, this.s, this.l * e, this.opacity);
  },
  darker(e) {
    return e = e == null ? Un : Math.pow(Un, e), new vt(this.h, this.s, this.l * e, this.opacity);
  },
  rgb() {
    var e = this.h % 360 + (this.h < 0) * 360, t = isNaN(e) || isNaN(this.s) ? 0 : this.s, r = this.l, n = r + (r < 0.5 ? r : 1 - r) * t, i = 2 * r - n;
    return new Ve(
      Cc(e >= 240 ? e - 240 : e + 120, i, n),
      Cc(e, i, n),
      Cc(e < 120 ? e + 240 : e - 120, i, n),
      this.opacity
    );
  },
  clamp() {
    return new vt(sm(this.h), Ri(this.s), Ri(this.l), oa(this.opacity));
  },
  displayable() {
    return (0 <= this.s && this.s <= 1 || isNaN(this.s)) && 0 <= this.l && this.l <= 1 && 0 <= this.opacity && this.opacity <= 1;
  },
  formatHsl() {
    const e = oa(this.opacity);
    return `${e === 1 ? "hsl(" : "hsla("}${sm(this.h)}, ${Ri(this.s) * 100}%, ${Ri(this.l) * 100}%${e === 1 ? ")" : `, ${e})`}`;
  }
}));
function sm(e) {
  return e = (e || 0) % 360, e < 0 ? e + 360 : e;
}
function Ri(e) {
  return Math.max(0, Math.min(1, e || 0));
}
function Cc(e, t, r) {
  return (e < 60 ? t + (r - t) * e / 60 : e < 180 ? r : e < 240 ? t + (r - t) * (240 - e) / 60 : t) * 255;
}
const Od = (e) => () => e;
function _E(e, t) {
  return function(r) {
    return e + r * t;
  };
}
function SE(e, t, r) {
  return e = Math.pow(e, r), t = Math.pow(t, r) - e, r = 1 / r, function(n) {
    return Math.pow(e + n * t, r);
  };
}
function PE(e) {
  return (e = +e) == 1 ? bx : function(t, r) {
    return r - t ? SE(t, r, e) : Od(isNaN(t) ? r : t);
  };
}
function bx(e, t) {
  var r = t - e;
  return r ? _E(e, r) : Od(isNaN(e) ? t : e);
}
const cm = function e(t) {
  var r = PE(t);
  function n(i, a) {
    var o = r((i = Zl(i)).r, (a = Zl(a)).r), u = r(i.g, a.g), s = r(i.b, a.b), c = bx(i.opacity, a.opacity);
    return function(f) {
      return i.r = o(f), i.g = u(f), i.b = s(f), i.opacity = c(f), i + "";
    };
  }
  return n.gamma = e, n;
}(1);
function AE(e, t) {
  t || (t = []);
  var r = e ? Math.min(t.length, e.length) : 0, n = t.slice(), i;
  return function(a) {
    for (i = 0; i < r; ++i) n[i] = e[i] * (1 - a) + t[i] * a;
    return n;
  };
}
function EE(e) {
  return ArrayBuffer.isView(e) && !(e instanceof DataView);
}
function TE(e, t) {
  var r = t ? t.length : 0, n = e ? Math.min(r, e.length) : 0, i = new Array(n), a = new Array(r), o;
  for (o = 0; o < n; ++o) i[o] = ln(e[o], t[o]);
  for (; o < r; ++o) a[o] = t[o];
  return function(u) {
    for (o = 0; o < n; ++o) a[o] = i[o](u);
    return a;
  };
}
function jE(e, t) {
  var r = /* @__PURE__ */ new Date();
  return e = +e, t = +t, function(n) {
    return r.setTime(e * (1 - n) + t * n), r;
  };
}
function ua(e, t) {
  return e = +e, t = +t, function(r) {
    return e * (1 - r) + t * r;
  };
}
function CE(e, t) {
  var r = {}, n = {}, i;
  (e === null || typeof e != "object") && (e = {}), (t === null || typeof t != "object") && (t = {});
  for (i in t)
    i in e ? r[i] = ln(e[i], t[i]) : n[i] = t[i];
  return function(a) {
    for (i in r) n[i] = r[i](a);
    return n;
  };
}
var Jl = /[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g, Mc = new RegExp(Jl.source, "g");
function ME(e) {
  return function() {
    return e;
  };
}
function IE(e) {
  return function(t) {
    return e(t) + "";
  };
}
function $E(e, t) {
  var r = Jl.lastIndex = Mc.lastIndex = 0, n, i, a, o = -1, u = [], s = [];
  for (e = e + "", t = t + ""; (n = Jl.exec(e)) && (i = Mc.exec(t)); )
    (a = i.index) > r && (a = t.slice(r, a), u[o] ? u[o] += a : u[++o] = a), (n = n[0]) === (i = i[0]) ? u[o] ? u[o] += i : u[++o] = i : (u[++o] = null, s.push({ i: o, x: ua(n, i) })), r = Mc.lastIndex;
  return r < t.length && (a = t.slice(r), u[o] ? u[o] += a : u[++o] = a), u.length < 2 ? s[0] ? IE(s[0].x) : ME(t) : (t = s.length, function(c) {
    for (var f = 0, l; f < t; ++f) u[(l = s[f]).i] = l.x(c);
    return u.join("");
  });
}
function ln(e, t) {
  var r = typeof t, n;
  return t == null || r === "boolean" ? Od(t) : (r === "number" ? ua : r === "string" ? (n = Gn(t)) ? (t = n, cm) : $E : t instanceof Gn ? cm : t instanceof Date ? jE : EE(t) ? AE : Array.isArray(t) ? TE : typeof t.valueOf != "function" && typeof t.toString != "function" || isNaN(t) ? CE : ua)(e, t);
}
function _d(e, t) {
  return e = +e, t = +t, function(r) {
    return Math.round(e * (1 - r) + t * r);
  };
}
function RE(e, t) {
  t === void 0 && (t = e, e = ln);
  for (var r = 0, n = t.length - 1, i = t[0], a = new Array(n < 0 ? 0 : n); r < n; ) a[r] = e(i, i = t[++r]);
  return function(o) {
    var u = Math.max(0, Math.min(n - 1, Math.floor(o *= n)));
    return a[u](o - u);
  };
}
function kE(e) {
  return function() {
    return e;
  };
}
function sa(e) {
  return +e;
}
var lm = [0, 1];
function Ke(e) {
  return e;
}
function Ql(e, t) {
  return (t -= e = +e) ? function(r) {
    return (r - e) / t;
  } : kE(isNaN(t) ? NaN : 0.5);
}
function NE(e, t) {
  var r;
  return e > t && (r = e, e = t, t = r), function(n) {
    return Math.max(e, Math.min(t, n));
  };
}
function DE(e, t, r) {
  var n = e[0], i = e[1], a = t[0], o = t[1];
  return i < n ? (n = Ql(i, n), a = r(o, a)) : (n = Ql(n, i), a = r(a, o)), function(u) {
    return a(n(u));
  };
}
function qE(e, t, r) {
  var n = Math.min(e.length, t.length) - 1, i = new Array(n), a = new Array(n), o = -1;
  for (e[n] < e[0] && (e = e.slice().reverse(), t = t.slice().reverse()); ++o < n; )
    i[o] = Ql(e[o], e[o + 1]), a[o] = r(t[o], t[o + 1]);
  return function(u) {
    var s = xi(e, u, 1, n) - 1;
    return a[s](i[s](u));
  };
}
function Oi(e, t) {
  return t.domain(e.domain()).range(e.range()).interpolate(e.interpolate()).clamp(e.clamp()).unknown(e.unknown());
}
function to() {
  var e = lm, t = lm, r = ln, n, i, a, o = Ke, u, s, c;
  function f() {
    var d = Math.min(e.length, t.length);
    return o !== Ke && (o = NE(e[0], e[d - 1])), u = d > 2 ? qE : DE, s = c = null, l;
  }
  function l(d) {
    return d == null || isNaN(d = +d) ? a : (s || (s = u(e.map(n), t, r)))(n(o(d)));
  }
  return l.invert = function(d) {
    return o(i((c || (c = u(t, e.map(n), ua)))(d)));
  }, l.domain = function(d) {
    return arguments.length ? (e = Array.from(d, sa), f()) : e.slice();
  }, l.range = function(d) {
    return arguments.length ? (t = Array.from(d), f()) : t.slice();
  }, l.rangeRound = function(d) {
    return t = Array.from(d), r = _d, f();
  }, l.clamp = function(d) {
    return arguments.length ? (o = d ? !0 : Ke, f()) : o !== Ke;
  }, l.interpolate = function(d) {
    return arguments.length ? (r = d, f()) : r;
  }, l.unknown = function(d) {
    return arguments.length ? (a = d, l) : a;
  }, function(d, p) {
    return n = d, i = p, f();
  };
}
function Sd() {
  return to()(Ke, Ke);
}
function LE(e) {
  return Math.abs(e = Math.round(e)) >= 1e21 ? e.toLocaleString("en").replace(/,/g, "") : e.toString(10);
}
function ca(e, t) {
  if ((r = (e = t ? e.toExponential(t - 1) : e.toExponential()).indexOf("e")) < 0) return null;
  var r, n = e.slice(0, r);
  return [
    n.length > 1 ? n[0] + n.slice(2) : n,
    +e.slice(r + 1)
  ];
}
function Fr(e) {
  return e = ca(Math.abs(e)), e ? e[1] : NaN;
}
function BE(e, t) {
  return function(r, n) {
    for (var i = r.length, a = [], o = 0, u = e[0], s = 0; i > 0 && u > 0 && (s + u + 1 > n && (u = Math.max(1, n - s)), a.push(r.substring(i -= u, i + u)), !((s += u + 1) > n)); )
      u = e[o = (o + 1) % e.length];
    return a.reverse().join(t);
  };
}
function FE(e) {
  return function(t) {
    return t.replace(/[0-9]/g, function(r) {
      return e[+r];
    });
  };
}
var zE = /^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;
function Hn(e) {
  if (!(t = zE.exec(e))) throw new Error("invalid format: " + e);
  var t;
  return new Pd({
    fill: t[1],
    align: t[2],
    sign: t[3],
    symbol: t[4],
    zero: t[5],
    width: t[6],
    comma: t[7],
    precision: t[8] && t[8].slice(1),
    trim: t[9],
    type: t[10]
  });
}
Hn.prototype = Pd.prototype;
function Pd(e) {
  this.fill = e.fill === void 0 ? " " : e.fill + "", this.align = e.align === void 0 ? ">" : e.align + "", this.sign = e.sign === void 0 ? "-" : e.sign + "", this.symbol = e.symbol === void 0 ? "" : e.symbol + "", this.zero = !!e.zero, this.width = e.width === void 0 ? void 0 : +e.width, this.comma = !!e.comma, this.precision = e.precision === void 0 ? void 0 : +e.precision, this.trim = !!e.trim, this.type = e.type === void 0 ? "" : e.type + "";
}
Pd.prototype.toString = function() {
  return this.fill + this.align + this.sign + this.symbol + (this.zero ? "0" : "") + (this.width === void 0 ? "" : Math.max(1, this.width | 0)) + (this.comma ? "," : "") + (this.precision === void 0 ? "" : "." + Math.max(0, this.precision | 0)) + (this.trim ? "~" : "") + this.type;
};
function UE(e) {
  e: for (var t = e.length, r = 1, n = -1, i; r < t; ++r)
    switch (e[r]) {
      case ".":
        n = i = r;
        break;
      case "0":
        n === 0 && (n = r), i = r;
        break;
      default:
        if (!+e[r]) break e;
        n > 0 && (n = 0);
        break;
    }
  return n > 0 ? e.slice(0, n) + e.slice(i + 1) : e;
}
var xx;
function WE(e, t) {
  var r = ca(e, t);
  if (!r) return e + "";
  var n = r[0], i = r[1], a = i - (xx = Math.max(-8, Math.min(8, Math.floor(i / 3))) * 3) + 1, o = n.length;
  return a === o ? n : a > o ? n + new Array(a - o + 1).join("0") : a > 0 ? n.slice(0, a) + "." + n.slice(a) : "0." + new Array(1 - a).join("0") + ca(e, Math.max(0, t + a - 1))[0];
}
function fm(e, t) {
  var r = ca(e, t);
  if (!r) return e + "";
  var n = r[0], i = r[1];
  return i < 0 ? "0." + new Array(-i).join("0") + n : n.length > i + 1 ? n.slice(0, i + 1) + "." + n.slice(i + 1) : n + new Array(i - n.length + 2).join("0");
}
const dm = {
  "%": (e, t) => (e * 100).toFixed(t),
  b: (e) => Math.round(e).toString(2),
  c: (e) => e + "",
  d: LE,
  e: (e, t) => e.toExponential(t),
  f: (e, t) => e.toFixed(t),
  g: (e, t) => e.toPrecision(t),
  o: (e) => Math.round(e).toString(8),
  p: (e, t) => fm(e * 100, t),
  r: fm,
  s: WE,
  X: (e) => Math.round(e).toString(16).toUpperCase(),
  x: (e) => Math.round(e).toString(16)
};
function pm(e) {
  return e;
}
var hm = Array.prototype.map, vm = ["y", "z", "a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"];
function GE(e) {
  var t = e.grouping === void 0 || e.thousands === void 0 ? pm : BE(hm.call(e.grouping, Number), e.thousands + ""), r = e.currency === void 0 ? "" : e.currency[0] + "", n = e.currency === void 0 ? "" : e.currency[1] + "", i = e.decimal === void 0 ? "." : e.decimal + "", a = e.numerals === void 0 ? pm : FE(hm.call(e.numerals, String)), o = e.percent === void 0 ? "%" : e.percent + "", u = e.minus === void 0 ? "−" : e.minus + "", s = e.nan === void 0 ? "NaN" : e.nan + "";
  function c(l) {
    l = Hn(l);
    var d = l.fill, p = l.align, y = l.sign, v = l.symbol, h = l.zero, g = l.width, w = l.comma, b = l.precision, O = l.trim, m = l.type;
    m === "n" ? (w = !0, m = "g") : dm[m] || (b === void 0 && (b = 12), O = !0, m = "g"), (h || d === "0" && p === "=") && (h = !0, d = "0", p = "=");
    var x = v === "$" ? r : v === "#" && /[boxX]/.test(m) ? "0" + m.toLowerCase() : "", _ = v === "$" ? n : /[%p]/.test(m) ? o : "", P = dm[m], E = /[defgprs%]/.test(m);
    b = b === void 0 ? 6 : /[gprs]/.test(m) ? Math.max(1, Math.min(21, b)) : Math.max(0, Math.min(20, b));
    function I(S) {
      var j = x, M = _, R, k, q;
      if (m === "c")
        M = P(S) + M, S = "";
      else {
        S = +S;
        var L = S < 0 || 1 / S < 0;
        if (S = isNaN(S) ? s : P(Math.abs(S), b), O && (S = UE(S)), L && +S == 0 && y !== "+" && (L = !1), j = (L ? y === "(" ? y : u : y === "-" || y === "(" ? "" : y) + j, M = (m === "s" ? vm[8 + xx / 3] : "") + M + (L && y === "(" ? ")" : ""), E) {
          for (R = -1, k = S.length; ++R < k; )
            if (q = S.charCodeAt(R), 48 > q || q > 57) {
              M = (q === 46 ? i + S.slice(R + 1) : S.slice(R)) + M, S = S.slice(0, R);
              break;
            }
        }
      }
      w && !h && (S = t(S, 1 / 0));
      var U = j.length + S.length + M.length, z = U < g ? new Array(g - U + 1).join(d) : "";
      switch (w && h && (S = t(z + S, z.length ? g - M.length : 1 / 0), z = ""), p) {
        case "<":
          S = j + S + M + z;
          break;
        case "=":
          S = j + z + S + M;
          break;
        case "^":
          S = z.slice(0, U = z.length >> 1) + j + S + M + z.slice(U);
          break;
        default:
          S = z + j + S + M;
          break;
      }
      return a(S);
    }
    return I.toString = function() {
      return l + "";
    }, I;
  }
  function f(l, d) {
    var p = c((l = Hn(l), l.type = "f", l)), y = Math.max(-8, Math.min(8, Math.floor(Fr(d) / 3))) * 3, v = Math.pow(10, -y), h = vm[8 + y / 3];
    return function(g) {
      return p(v * g) + h;
    };
  }
  return {
    format: c,
    formatPrefix: f
  };
}
var ki, Ad, wx;
HE({
  thousands: ",",
  grouping: [3],
  currency: ["$", ""]
});
function HE(e) {
  return ki = GE(e), Ad = ki.format, wx = ki.formatPrefix, ki;
}
function KE(e) {
  return Math.max(0, -Fr(Math.abs(e)));
}
function VE(e, t) {
  return Math.max(0, Math.max(-8, Math.min(8, Math.floor(Fr(t) / 3))) * 3 - Fr(Math.abs(e)));
}
function YE(e, t) {
  return e = Math.abs(e), t = Math.abs(t) - e, Math.max(0, Fr(t) - Fr(e)) + 1;
}
function Ox(e, t, r, n) {
  var i = Yl(e, t, r), a;
  switch (n = Hn(n ?? ",f"), n.type) {
    case "s": {
      var o = Math.max(Math.abs(e), Math.abs(t));
      return n.precision == null && !isNaN(a = VE(i, o)) && (n.precision = a), wx(n, o);
    }
    case "":
    case "e":
    case "g":
    case "p":
    case "r": {
      n.precision == null && !isNaN(a = YE(i, Math.max(Math.abs(e), Math.abs(t)))) && (n.precision = a - (n.type === "e"));
      break;
    }
    case "f":
    case "%": {
      n.precision == null && !isNaN(a = KE(i)) && (n.precision = a - (n.type === "%") * 2);
      break;
    }
  }
  return Ad(n);
}
function Qt(e) {
  var t = e.domain;
  return e.ticks = function(r) {
    var n = t();
    return Kl(n[0], n[n.length - 1], r ?? 10);
  }, e.tickFormat = function(r, n) {
    var i = t();
    return Ox(i[0], i[i.length - 1], r ?? 10, n);
  }, e.nice = function(r) {
    r == null && (r = 10);
    var n = t(), i = 0, a = n.length - 1, o = n[i], u = n[a], s, c, f = 10;
    for (u < o && (c = o, o = u, u = c, c = i, i = a, a = c); f-- > 0; ) {
      if (c = Vl(o, u, r), c === s)
        return n[i] = o, n[a] = u, t(n);
      if (c > 0)
        o = Math.floor(o / c) * c, u = Math.ceil(u / c) * c;
      else if (c < 0)
        o = Math.ceil(o * c) / c, u = Math.floor(u * c) / c;
      else
        break;
      s = c;
    }
    return e;
  }, e;
}
function la() {
  var e = Sd();
  return e.copy = function() {
    return Oi(e, la());
  }, st.apply(e, arguments), Qt(e);
}
function _x(e) {
  var t;
  function r(n) {
    return n == null || isNaN(n = +n) ? t : n;
  }
  return r.invert = r, r.domain = r.range = function(n) {
    return arguments.length ? (e = Array.from(n, sa), r) : e.slice();
  }, r.unknown = function(n) {
    return arguments.length ? (t = n, r) : t;
  }, r.copy = function() {
    return _x(e).unknown(t);
  }, e = arguments.length ? Array.from(e, sa) : [0, 1], Qt(r);
}
function Sx(e, t) {
  e = e.slice();
  var r = 0, n = e.length - 1, i = e[r], a = e[n], o;
  return a < i && (o = r, r = n, n = o, o = i, i = a, a = o), e[r] = t.floor(i), e[n] = t.ceil(a), e;
}
function ym(e) {
  return Math.log(e);
}
function mm(e) {
  return Math.exp(e);
}
function XE(e) {
  return -Math.log(-e);
}
function ZE(e) {
  return -Math.exp(-e);
}
function JE(e) {
  return isFinite(e) ? +("1e" + e) : e < 0 ? 0 : e;
}
function QE(e) {
  return e === 10 ? JE : e === Math.E ? Math.exp : (t) => Math.pow(e, t);
}
function eT(e) {
  return e === Math.E ? Math.log : e === 10 && Math.log10 || e === 2 && Math.log2 || (e = Math.log(e), (t) => Math.log(t) / e);
}
function gm(e) {
  return (t, r) => -e(-t, r);
}
function Ed(e) {
  const t = e(ym, mm), r = t.domain;
  let n = 10, i, a;
  function o() {
    return i = eT(n), a = QE(n), r()[0] < 0 ? (i = gm(i), a = gm(a), e(XE, ZE)) : e(ym, mm), t;
  }
  return t.base = function(u) {
    return arguments.length ? (n = +u, o()) : n;
  }, t.domain = function(u) {
    return arguments.length ? (r(u), o()) : r();
  }, t.ticks = (u) => {
    const s = r();
    let c = s[0], f = s[s.length - 1];
    const l = f < c;
    l && ([c, f] = [f, c]);
    let d = i(c), p = i(f), y, v;
    const h = u == null ? 10 : +u;
    let g = [];
    if (!(n % 1) && p - d < h) {
      if (d = Math.floor(d), p = Math.ceil(p), c > 0) {
        for (; d <= p; ++d)
          for (y = 1; y < n; ++y)
            if (v = d < 0 ? y / a(-d) : y * a(d), !(v < c)) {
              if (v > f) break;
              g.push(v);
            }
      } else for (; d <= p; ++d)
        for (y = n - 1; y >= 1; --y)
          if (v = d > 0 ? y / a(-d) : y * a(d), !(v < c)) {
            if (v > f) break;
            g.push(v);
          }
      g.length * 2 < h && (g = Kl(c, f, h));
    } else
      g = Kl(d, p, Math.min(p - d, h)).map(a);
    return l ? g.reverse() : g;
  }, t.tickFormat = (u, s) => {
    if (u == null && (u = 10), s == null && (s = n === 10 ? "s" : ","), typeof s != "function" && (!(n % 1) && (s = Hn(s)).precision == null && (s.trim = !0), s = Ad(s)), u === 1 / 0) return s;
    const c = Math.max(1, n * u / t.ticks().length);
    return (f) => {
      let l = f / a(Math.round(i(f)));
      return l * n < n - 0.5 && (l *= n), l <= c ? s(f) : "";
    };
  }, t.nice = () => r(Sx(r(), {
    floor: (u) => a(Math.floor(i(u))),
    ceil: (u) => a(Math.ceil(i(u)))
  })), t;
}
function Px() {
  const e = Ed(to()).domain([1, 10]);
  return e.copy = () => Oi(e, Px()).base(e.base()), st.apply(e, arguments), e;
}
function bm(e) {
  return function(t) {
    return Math.sign(t) * Math.log1p(Math.abs(t / e));
  };
}
function xm(e) {
  return function(t) {
    return Math.sign(t) * Math.expm1(Math.abs(t)) * e;
  };
}
function Td(e) {
  var t = 1, r = e(bm(t), xm(t));
  return r.constant = function(n) {
    return arguments.length ? e(bm(t = +n), xm(t)) : t;
  }, Qt(r);
}
function Ax() {
  var e = Td(to());
  return e.copy = function() {
    return Oi(e, Ax()).constant(e.constant());
  }, st.apply(e, arguments);
}
function wm(e) {
  return function(t) {
    return t < 0 ? -Math.pow(-t, e) : Math.pow(t, e);
  };
}
function tT(e) {
  return e < 0 ? -Math.sqrt(-e) : Math.sqrt(e);
}
function rT(e) {
  return e < 0 ? -e * e : e * e;
}
function jd(e) {
  var t = e(Ke, Ke), r = 1;
  function n() {
    return r === 1 ? e(Ke, Ke) : r === 0.5 ? e(tT, rT) : e(wm(r), wm(1 / r));
  }
  return t.exponent = function(i) {
    return arguments.length ? (r = +i, n()) : r;
  }, Qt(t);
}
function Cd() {
  var e = jd(to());
  return e.copy = function() {
    return Oi(e, Cd()).exponent(e.exponent());
  }, st.apply(e, arguments), e;
}
function nT() {
  return Cd.apply(null, arguments).exponent(0.5);
}
function Om(e) {
  return Math.sign(e) * e * e;
}
function iT(e) {
  return Math.sign(e) * Math.sqrt(Math.abs(e));
}
function Ex() {
  var e = Sd(), t = [0, 1], r = !1, n;
  function i(a) {
    var o = iT(e(a));
    return isNaN(o) ? n : r ? Math.round(o) : o;
  }
  return i.invert = function(a) {
    return e.invert(Om(a));
  }, i.domain = function(a) {
    return arguments.length ? (e.domain(a), i) : e.domain();
  }, i.range = function(a) {
    return arguments.length ? (e.range((t = Array.from(a, sa)).map(Om)), i) : t.slice();
  }, i.rangeRound = function(a) {
    return i.range(a).round(!0);
  }, i.round = function(a) {
    return arguments.length ? (r = !!a, i) : r;
  }, i.clamp = function(a) {
    return arguments.length ? (e.clamp(a), i) : e.clamp();
  }, i.unknown = function(a) {
    return arguments.length ? (n = a, i) : n;
  }, i.copy = function() {
    return Ex(e.domain(), t).round(r).clamp(e.clamp()).unknown(n);
  }, st.apply(i, arguments), Qt(i);
}
function Tx() {
  var e = [], t = [], r = [], n;
  function i() {
    var o = 0, u = Math.max(1, t.length);
    for (r = new Array(u - 1); ++o < u; ) r[o - 1] = cE(e, o / u);
    return a;
  }
  function a(o) {
    return o == null || isNaN(o = +o) ? n : t[xi(r, o)];
  }
  return a.invertExtent = function(o) {
    var u = t.indexOf(o);
    return u < 0 ? [NaN, NaN] : [
      u > 0 ? r[u - 1] : e[0],
      u < r.length ? r[u] : e[e.length - 1]
    ];
  }, a.domain = function(o) {
    if (!arguments.length) return e.slice();
    e = [];
    for (let u of o) u != null && !isNaN(u = +u) && e.push(u);
    return e.sort(Kt), i();
  }, a.range = function(o) {
    return arguments.length ? (t = Array.from(o), i()) : t.slice();
  }, a.unknown = function(o) {
    return arguments.length ? (n = o, a) : n;
  }, a.quantiles = function() {
    return r.slice();
  }, a.copy = function() {
    return Tx().domain(e).range(t).unknown(n);
  }, st.apply(a, arguments);
}
function jx() {
  var e = 0, t = 1, r = 1, n = [0.5], i = [0, 1], a;
  function o(s) {
    return s != null && s <= s ? i[xi(n, s, 0, r)] : a;
  }
  function u() {
    var s = -1;
    for (n = new Array(r); ++s < r; ) n[s] = ((s + 1) * t - (s - r) * e) / (r + 1);
    return o;
  }
  return o.domain = function(s) {
    return arguments.length ? ([e, t] = s, e = +e, t = +t, u()) : [e, t];
  }, o.range = function(s) {
    return arguments.length ? (r = (i = Array.from(s)).length - 1, u()) : i.slice();
  }, o.invertExtent = function(s) {
    var c = i.indexOf(s);
    return c < 0 ? [NaN, NaN] : c < 1 ? [e, n[0]] : c >= r ? [n[r - 1], t] : [n[c - 1], n[c]];
  }, o.unknown = function(s) {
    return arguments.length && (a = s), o;
  }, o.thresholds = function() {
    return n.slice();
  }, o.copy = function() {
    return jx().domain([e, t]).range(i).unknown(a);
  }, st.apply(Qt(o), arguments);
}
function Cx() {
  var e = [0.5], t = [0, 1], r, n = 1;
  function i(a) {
    return a != null && a <= a ? t[xi(e, a, 0, n)] : r;
  }
  return i.domain = function(a) {
    return arguments.length ? (e = Array.from(a), n = Math.min(e.length, t.length - 1), i) : e.slice();
  }, i.range = function(a) {
    return arguments.length ? (t = Array.from(a), n = Math.min(e.length, t.length - 1), i) : t.slice();
  }, i.invertExtent = function(a) {
    var o = t.indexOf(a);
    return [e[o - 1], e[o]];
  }, i.unknown = function(a) {
    return arguments.length ? (r = a, i) : r;
  }, i.copy = function() {
    return Cx().domain(e).range(t).unknown(r);
  }, st.apply(i, arguments);
}
const Ic = /* @__PURE__ */ new Date(), $c = /* @__PURE__ */ new Date();
function Ne(e, t, r, n) {
  function i(a) {
    return e(a = arguments.length === 0 ? /* @__PURE__ */ new Date() : /* @__PURE__ */ new Date(+a)), a;
  }
  return i.floor = (a) => (e(a = /* @__PURE__ */ new Date(+a)), a), i.ceil = (a) => (e(a = new Date(a - 1)), t(a, 1), e(a), a), i.round = (a) => {
    const o = i(a), u = i.ceil(a);
    return a - o < u - a ? o : u;
  }, i.offset = (a, o) => (t(a = /* @__PURE__ */ new Date(+a), o == null ? 1 : Math.floor(o)), a), i.range = (a, o, u) => {
    const s = [];
    if (a = i.ceil(a), u = u == null ? 1 : Math.floor(u), !(a < o) || !(u > 0)) return s;
    let c;
    do
      s.push(c = /* @__PURE__ */ new Date(+a)), t(a, u), e(a);
    while (c < a && a < o);
    return s;
  }, i.filter = (a) => Ne((o) => {
    if (o >= o) for (; e(o), !a(o); ) o.setTime(o - 1);
  }, (o, u) => {
    if (o >= o)
      if (u < 0) for (; ++u <= 0; )
        for (; t(o, -1), !a(o); )
          ;
      else for (; --u >= 0; )
        for (; t(o, 1), !a(o); )
          ;
  }), r && (i.count = (a, o) => (Ic.setTime(+a), $c.setTime(+o), e(Ic), e($c), Math.floor(r(Ic, $c))), i.every = (a) => (a = Math.floor(a), !isFinite(a) || !(a > 0) ? null : a > 1 ? i.filter(n ? (o) => n(o) % a === 0 : (o) => i.count(0, o) % a === 0) : i)), i;
}
const fa = Ne(() => {
}, (e, t) => {
  e.setTime(+e + t);
}, (e, t) => t - e);
fa.every = (e) => (e = Math.floor(e), !isFinite(e) || !(e > 0) ? null : e > 1 ? Ne((t) => {
  t.setTime(Math.floor(t / e) * e);
}, (t, r) => {
  t.setTime(+t + r * e);
}, (t, r) => (r - t) / e) : fa);
fa.range;
const jt = 1e3, it = jt * 60, Ct = it * 60, Rt = Ct * 24, Md = Rt * 7, _m = Rt * 30, Rc = Rt * 365, sr = Ne((e) => {
  e.setTime(e - e.getMilliseconds());
}, (e, t) => {
  e.setTime(+e + t * jt);
}, (e, t) => (t - e) / jt, (e) => e.getUTCSeconds());
sr.range;
const Id = Ne((e) => {
  e.setTime(e - e.getMilliseconds() - e.getSeconds() * jt);
}, (e, t) => {
  e.setTime(+e + t * it);
}, (e, t) => (t - e) / it, (e) => e.getMinutes());
Id.range;
const $d = Ne((e) => {
  e.setUTCSeconds(0, 0);
}, (e, t) => {
  e.setTime(+e + t * it);
}, (e, t) => (t - e) / it, (e) => e.getUTCMinutes());
$d.range;
const Rd = Ne((e) => {
  e.setTime(e - e.getMilliseconds() - e.getSeconds() * jt - e.getMinutes() * it);
}, (e, t) => {
  e.setTime(+e + t * Ct);
}, (e, t) => (t - e) / Ct, (e) => e.getHours());
Rd.range;
const kd = Ne((e) => {
  e.setUTCMinutes(0, 0, 0);
}, (e, t) => {
  e.setTime(+e + t * Ct);
}, (e, t) => (t - e) / Ct, (e) => e.getUTCHours());
kd.range;
const _i = Ne(
  (e) => e.setHours(0, 0, 0, 0),
  (e, t) => e.setDate(e.getDate() + t),
  (e, t) => (t - e - (t.getTimezoneOffset() - e.getTimezoneOffset()) * it) / Rt,
  (e) => e.getDate() - 1
);
_i.range;
const ro = Ne((e) => {
  e.setUTCHours(0, 0, 0, 0);
}, (e, t) => {
  e.setUTCDate(e.getUTCDate() + t);
}, (e, t) => (t - e) / Rt, (e) => e.getUTCDate() - 1);
ro.range;
const Mx = Ne((e) => {
  e.setUTCHours(0, 0, 0, 0);
}, (e, t) => {
  e.setUTCDate(e.getUTCDate() + t);
}, (e, t) => (t - e) / Rt, (e) => Math.floor(e / Rt));
Mx.range;
function br(e) {
  return Ne((t) => {
    t.setDate(t.getDate() - (t.getDay() + 7 - e) % 7), t.setHours(0, 0, 0, 0);
  }, (t, r) => {
    t.setDate(t.getDate() + r * 7);
  }, (t, r) => (r - t - (r.getTimezoneOffset() - t.getTimezoneOffset()) * it) / Md);
}
const no = br(0), da = br(1), aT = br(2), oT = br(3), zr = br(4), uT = br(5), sT = br(6);
no.range;
da.range;
aT.range;
oT.range;
zr.range;
uT.range;
sT.range;
function xr(e) {
  return Ne((t) => {
    t.setUTCDate(t.getUTCDate() - (t.getUTCDay() + 7 - e) % 7), t.setUTCHours(0, 0, 0, 0);
  }, (t, r) => {
    t.setUTCDate(t.getUTCDate() + r * 7);
  }, (t, r) => (r - t) / Md);
}
const io = xr(0), pa = xr(1), cT = xr(2), lT = xr(3), Ur = xr(4), fT = xr(5), dT = xr(6);
io.range;
pa.range;
cT.range;
lT.range;
Ur.range;
fT.range;
dT.range;
const Nd = Ne((e) => {
  e.setDate(1), e.setHours(0, 0, 0, 0);
}, (e, t) => {
  e.setMonth(e.getMonth() + t);
}, (e, t) => t.getMonth() - e.getMonth() + (t.getFullYear() - e.getFullYear()) * 12, (e) => e.getMonth());
Nd.range;
const Dd = Ne((e) => {
  e.setUTCDate(1), e.setUTCHours(0, 0, 0, 0);
}, (e, t) => {
  e.setUTCMonth(e.getUTCMonth() + t);
}, (e, t) => t.getUTCMonth() - e.getUTCMonth() + (t.getUTCFullYear() - e.getUTCFullYear()) * 12, (e) => e.getUTCMonth());
Dd.range;
const kt = Ne((e) => {
  e.setMonth(0, 1), e.setHours(0, 0, 0, 0);
}, (e, t) => {
  e.setFullYear(e.getFullYear() + t);
}, (e, t) => t.getFullYear() - e.getFullYear(), (e) => e.getFullYear());
kt.every = (e) => !isFinite(e = Math.floor(e)) || !(e > 0) ? null : Ne((t) => {
  t.setFullYear(Math.floor(t.getFullYear() / e) * e), t.setMonth(0, 1), t.setHours(0, 0, 0, 0);
}, (t, r) => {
  t.setFullYear(t.getFullYear() + r * e);
});
kt.range;
const Nt = Ne((e) => {
  e.setUTCMonth(0, 1), e.setUTCHours(0, 0, 0, 0);
}, (e, t) => {
  e.setUTCFullYear(e.getUTCFullYear() + t);
}, (e, t) => t.getUTCFullYear() - e.getUTCFullYear(), (e) => e.getUTCFullYear());
Nt.every = (e) => !isFinite(e = Math.floor(e)) || !(e > 0) ? null : Ne((t) => {
  t.setUTCFullYear(Math.floor(t.getUTCFullYear() / e) * e), t.setUTCMonth(0, 1), t.setUTCHours(0, 0, 0, 0);
}, (t, r) => {
  t.setUTCFullYear(t.getUTCFullYear() + r * e);
});
Nt.range;
function Ix(e, t, r, n, i, a) {
  const o = [
    [sr, 1, jt],
    [sr, 5, 5 * jt],
    [sr, 15, 15 * jt],
    [sr, 30, 30 * jt],
    [a, 1, it],
    [a, 5, 5 * it],
    [a, 15, 15 * it],
    [a, 30, 30 * it],
    [i, 1, Ct],
    [i, 3, 3 * Ct],
    [i, 6, 6 * Ct],
    [i, 12, 12 * Ct],
    [n, 1, Rt],
    [n, 2, 2 * Rt],
    [r, 1, Md],
    [t, 1, _m],
    [t, 3, 3 * _m],
    [e, 1, Rc]
  ];
  function u(c, f, l) {
    const d = f < c;
    d && ([c, f] = [f, c]);
    const p = l && typeof l.range == "function" ? l : s(c, f, l), y = p ? p.range(c, +f + 1) : [];
    return d ? y.reverse() : y;
  }
  function s(c, f, l) {
    const d = Math.abs(f - c) / l, p = bd(([, , h]) => h).right(o, d);
    if (p === o.length) return e.every(Yl(c / Rc, f / Rc, l));
    if (p === 0) return fa.every(Math.max(Yl(c, f, l), 1));
    const [y, v] = o[d / o[p - 1][2] < o[p][2] / d ? p - 1 : p];
    return y.every(v);
  }
  return [u, s];
}
const [pT, hT] = Ix(Nt, Dd, io, Mx, kd, $d), [vT, yT] = Ix(kt, Nd, no, _i, Rd, Id);
function kc(e) {
  if (0 <= e.y && e.y < 100) {
    var t = new Date(-1, e.m, e.d, e.H, e.M, e.S, e.L);
    return t.setFullYear(e.y), t;
  }
  return new Date(e.y, e.m, e.d, e.H, e.M, e.S, e.L);
}
function Nc(e) {
  if (0 <= e.y && e.y < 100) {
    var t = new Date(Date.UTC(-1, e.m, e.d, e.H, e.M, e.S, e.L));
    return t.setUTCFullYear(e.y), t;
  }
  return new Date(Date.UTC(e.y, e.m, e.d, e.H, e.M, e.S, e.L));
}
function wn(e, t, r) {
  return { y: e, m: t, d: r, H: 0, M: 0, S: 0, L: 0 };
}
function mT(e) {
  var t = e.dateTime, r = e.date, n = e.time, i = e.periods, a = e.days, o = e.shortDays, u = e.months, s = e.shortMonths, c = On(i), f = _n(i), l = On(a), d = _n(a), p = On(o), y = _n(o), v = On(u), h = _n(u), g = On(s), w = _n(s), b = {
    a: L,
    A: U,
    b: z,
    B: $,
    c: null,
    d: jm,
    e: jm,
    f: BT,
    g: XT,
    G: JT,
    H: DT,
    I: qT,
    j: LT,
    L: $x,
    m: FT,
    M: zT,
    p: D,
    q: B,
    Q: Im,
    s: $m,
    S: UT,
    u: WT,
    U: GT,
    V: HT,
    w: KT,
    W: VT,
    x: null,
    X: null,
    y: YT,
    Y: ZT,
    Z: QT,
    "%": Mm
  }, O = {
    a: G,
    A: V,
    b: te,
    B: re,
    c: null,
    d: Cm,
    e: Cm,
    f: nj,
    g: pj,
    G: vj,
    H: ej,
    I: tj,
    j: rj,
    L: kx,
    m: ij,
    M: aj,
    p: ae,
    q: ne,
    Q: Im,
    s: $m,
    S: oj,
    u: uj,
    U: sj,
    V: cj,
    w: lj,
    W: fj,
    x: null,
    X: null,
    y: dj,
    Y: hj,
    Z: yj,
    "%": Mm
  }, m = {
    a: I,
    A: S,
    b: j,
    B: M,
    c: R,
    d: Em,
    e: Em,
    f: $T,
    g: Am,
    G: Pm,
    H: Tm,
    I: Tm,
    j: jT,
    L: IT,
    m: TT,
    M: CT,
    p: E,
    q: ET,
    Q: kT,
    s: NT,
    S: MT,
    u: OT,
    U: _T,
    V: ST,
    w: wT,
    W: PT,
    x: k,
    X: q,
    y: Am,
    Y: Pm,
    Z: AT,
    "%": RT
  };
  b.x = x(r, b), b.X = x(n, b), b.c = x(t, b), O.x = x(r, O), O.X = x(n, O), O.c = x(t, O);
  function x(F, H) {
    return function(ee) {
      var C = [], se = -1, W = 0, he = F.length, Oe, Ce, ct;
      for (ee instanceof Date || (ee = /* @__PURE__ */ new Date(+ee)); ++se < he; )
        F.charCodeAt(se) === 37 && (C.push(F.slice(W, se)), (Ce = Sm[Oe = F.charAt(++se)]) != null ? Oe = F.charAt(++se) : Ce = Oe === "e" ? " " : "0", (ct = H[Oe]) && (Oe = ct(ee, Ce)), C.push(Oe), W = se + 1);
      return C.push(F.slice(W, se)), C.join("");
    };
  }
  function _(F, H) {
    return function(ee) {
      var C = wn(1900, void 0, 1), se = P(C, F, ee += "", 0), W, he;
      if (se != ee.length) return null;
      if ("Q" in C) return new Date(C.Q);
      if ("s" in C) return new Date(C.s * 1e3 + ("L" in C ? C.L : 0));
      if (H && !("Z" in C) && (C.Z = 0), "p" in C && (C.H = C.H % 12 + C.p * 12), C.m === void 0 && (C.m = "q" in C ? C.q : 0), "V" in C) {
        if (C.V < 1 || C.V > 53) return null;
        "w" in C || (C.w = 1), "Z" in C ? (W = Nc(wn(C.y, 0, 1)), he = W.getUTCDay(), W = he > 4 || he === 0 ? pa.ceil(W) : pa(W), W = ro.offset(W, (C.V - 1) * 7), C.y = W.getUTCFullYear(), C.m = W.getUTCMonth(), C.d = W.getUTCDate() + (C.w + 6) % 7) : (W = kc(wn(C.y, 0, 1)), he = W.getDay(), W = he > 4 || he === 0 ? da.ceil(W) : da(W), W = _i.offset(W, (C.V - 1) * 7), C.y = W.getFullYear(), C.m = W.getMonth(), C.d = W.getDate() + (C.w + 6) % 7);
      } else ("W" in C || "U" in C) && ("w" in C || (C.w = "u" in C ? C.u % 7 : "W" in C ? 1 : 0), he = "Z" in C ? Nc(wn(C.y, 0, 1)).getUTCDay() : kc(wn(C.y, 0, 1)).getDay(), C.m = 0, C.d = "W" in C ? (C.w + 6) % 7 + C.W * 7 - (he + 5) % 7 : C.w + C.U * 7 - (he + 6) % 7);
      return "Z" in C ? (C.H += C.Z / 100 | 0, C.M += C.Z % 100, Nc(C)) : kc(C);
    };
  }
  function P(F, H, ee, C) {
    for (var se = 0, W = H.length, he = ee.length, Oe, Ce; se < W; ) {
      if (C >= he) return -1;
      if (Oe = H.charCodeAt(se++), Oe === 37) {
        if (Oe = H.charAt(se++), Ce = m[Oe in Sm ? H.charAt(se++) : Oe], !Ce || (C = Ce(F, ee, C)) < 0) return -1;
      } else if (Oe != ee.charCodeAt(C++))
        return -1;
    }
    return C;
  }
  function E(F, H, ee) {
    var C = c.exec(H.slice(ee));
    return C ? (F.p = f.get(C[0].toLowerCase()), ee + C[0].length) : -1;
  }
  function I(F, H, ee) {
    var C = p.exec(H.slice(ee));
    return C ? (F.w = y.get(C[0].toLowerCase()), ee + C[0].length) : -1;
  }
  function S(F, H, ee) {
    var C = l.exec(H.slice(ee));
    return C ? (F.w = d.get(C[0].toLowerCase()), ee + C[0].length) : -1;
  }
  function j(F, H, ee) {
    var C = g.exec(H.slice(ee));
    return C ? (F.m = w.get(C[0].toLowerCase()), ee + C[0].length) : -1;
  }
  function M(F, H, ee) {
    var C = v.exec(H.slice(ee));
    return C ? (F.m = h.get(C[0].toLowerCase()), ee + C[0].length) : -1;
  }
  function R(F, H, ee) {
    return P(F, t, H, ee);
  }
  function k(F, H, ee) {
    return P(F, r, H, ee);
  }
  function q(F, H, ee) {
    return P(F, n, H, ee);
  }
  function L(F) {
    return o[F.getDay()];
  }
  function U(F) {
    return a[F.getDay()];
  }
  function z(F) {
    return s[F.getMonth()];
  }
  function $(F) {
    return u[F.getMonth()];
  }
  function D(F) {
    return i[+(F.getHours() >= 12)];
  }
  function B(F) {
    return 1 + ~~(F.getMonth() / 3);
  }
  function G(F) {
    return o[F.getUTCDay()];
  }
  function V(F) {
    return a[F.getUTCDay()];
  }
  function te(F) {
    return s[F.getUTCMonth()];
  }
  function re(F) {
    return u[F.getUTCMonth()];
  }
  function ae(F) {
    return i[+(F.getUTCHours() >= 12)];
  }
  function ne(F) {
    return 1 + ~~(F.getUTCMonth() / 3);
  }
  return {
    format: function(F) {
      var H = x(F += "", b);
      return H.toString = function() {
        return F;
      }, H;
    },
    parse: function(F) {
      var H = _(F += "", !1);
      return H.toString = function() {
        return F;
      }, H;
    },
    utcFormat: function(F) {
      var H = x(F += "", O);
      return H.toString = function() {
        return F;
      }, H;
    },
    utcParse: function(F) {
      var H = _(F += "", !0);
      return H.toString = function() {
        return F;
      }, H;
    }
  };
}
var Sm = { "-": "", _: " ", 0: "0" }, Le = /^\s*\d+/, gT = /^%/, bT = /[\\^$*+?|[\]().{}]/g;
function de(e, t, r) {
  var n = e < 0 ? "-" : "", i = (n ? -e : e) + "", a = i.length;
  return n + (a < r ? new Array(r - a + 1).join(t) + i : i);
}
function xT(e) {
  return e.replace(bT, "\\$&");
}
function On(e) {
  return new RegExp("^(?:" + e.map(xT).join("|") + ")", "i");
}
function _n(e) {
  return new Map(e.map((t, r) => [t.toLowerCase(), r]));
}
function wT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 1));
  return n ? (e.w = +n[0], r + n[0].length) : -1;
}
function OT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 1));
  return n ? (e.u = +n[0], r + n[0].length) : -1;
}
function _T(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.U = +n[0], r + n[0].length) : -1;
}
function ST(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.V = +n[0], r + n[0].length) : -1;
}
function PT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.W = +n[0], r + n[0].length) : -1;
}
function Pm(e, t, r) {
  var n = Le.exec(t.slice(r, r + 4));
  return n ? (e.y = +n[0], r + n[0].length) : -1;
}
function Am(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.y = +n[0] + (+n[0] > 68 ? 1900 : 2e3), r + n[0].length) : -1;
}
function AT(e, t, r) {
  var n = /^(Z)|([+-]\d\d)(?::?(\d\d))?/.exec(t.slice(r, r + 6));
  return n ? (e.Z = n[1] ? 0 : -(n[2] + (n[3] || "00")), r + n[0].length) : -1;
}
function ET(e, t, r) {
  var n = Le.exec(t.slice(r, r + 1));
  return n ? (e.q = n[0] * 3 - 3, r + n[0].length) : -1;
}
function TT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.m = n[0] - 1, r + n[0].length) : -1;
}
function Em(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.d = +n[0], r + n[0].length) : -1;
}
function jT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 3));
  return n ? (e.m = 0, e.d = +n[0], r + n[0].length) : -1;
}
function Tm(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.H = +n[0], r + n[0].length) : -1;
}
function CT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.M = +n[0], r + n[0].length) : -1;
}
function MT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 2));
  return n ? (e.S = +n[0], r + n[0].length) : -1;
}
function IT(e, t, r) {
  var n = Le.exec(t.slice(r, r + 3));
  return n ? (e.L = +n[0], r + n[0].length) : -1;
}
function $T(e, t, r) {
  var n = Le.exec(t.slice(r, r + 6));
  return n ? (e.L = Math.floor(n[0] / 1e3), r + n[0].length) : -1;
}
function RT(e, t, r) {
  var n = gT.exec(t.slice(r, r + 1));
  return n ? r + n[0].length : -1;
}
function kT(e, t, r) {
  var n = Le.exec(t.slice(r));
  return n ? (e.Q = +n[0], r + n[0].length) : -1;
}
function NT(e, t, r) {
  var n = Le.exec(t.slice(r));
  return n ? (e.s = +n[0], r + n[0].length) : -1;
}
function jm(e, t) {
  return de(e.getDate(), t, 2);
}
function DT(e, t) {
  return de(e.getHours(), t, 2);
}
function qT(e, t) {
  return de(e.getHours() % 12 || 12, t, 2);
}
function LT(e, t) {
  return de(1 + _i.count(kt(e), e), t, 3);
}
function $x(e, t) {
  return de(e.getMilliseconds(), t, 3);
}
function BT(e, t) {
  return $x(e, t) + "000";
}
function FT(e, t) {
  return de(e.getMonth() + 1, t, 2);
}
function zT(e, t) {
  return de(e.getMinutes(), t, 2);
}
function UT(e, t) {
  return de(e.getSeconds(), t, 2);
}
function WT(e) {
  var t = e.getDay();
  return t === 0 ? 7 : t;
}
function GT(e, t) {
  return de(no.count(kt(e) - 1, e), t, 2);
}
function Rx(e) {
  var t = e.getDay();
  return t >= 4 || t === 0 ? zr(e) : zr.ceil(e);
}
function HT(e, t) {
  return e = Rx(e), de(zr.count(kt(e), e) + (kt(e).getDay() === 4), t, 2);
}
function KT(e) {
  return e.getDay();
}
function VT(e, t) {
  return de(da.count(kt(e) - 1, e), t, 2);
}
function YT(e, t) {
  return de(e.getFullYear() % 100, t, 2);
}
function XT(e, t) {
  return e = Rx(e), de(e.getFullYear() % 100, t, 2);
}
function ZT(e, t) {
  return de(e.getFullYear() % 1e4, t, 4);
}
function JT(e, t) {
  var r = e.getDay();
  return e = r >= 4 || r === 0 ? zr(e) : zr.ceil(e), de(e.getFullYear() % 1e4, t, 4);
}
function QT(e) {
  var t = e.getTimezoneOffset();
  return (t > 0 ? "-" : (t *= -1, "+")) + de(t / 60 | 0, "0", 2) + de(t % 60, "0", 2);
}
function Cm(e, t) {
  return de(e.getUTCDate(), t, 2);
}
function ej(e, t) {
  return de(e.getUTCHours(), t, 2);
}
function tj(e, t) {
  return de(e.getUTCHours() % 12 || 12, t, 2);
}
function rj(e, t) {
  return de(1 + ro.count(Nt(e), e), t, 3);
}
function kx(e, t) {
  return de(e.getUTCMilliseconds(), t, 3);
}
function nj(e, t) {
  return kx(e, t) + "000";
}
function ij(e, t) {
  return de(e.getUTCMonth() + 1, t, 2);
}
function aj(e, t) {
  return de(e.getUTCMinutes(), t, 2);
}
function oj(e, t) {
  return de(e.getUTCSeconds(), t, 2);
}
function uj(e) {
  var t = e.getUTCDay();
  return t === 0 ? 7 : t;
}
function sj(e, t) {
  return de(io.count(Nt(e) - 1, e), t, 2);
}
function Nx(e) {
  var t = e.getUTCDay();
  return t >= 4 || t === 0 ? Ur(e) : Ur.ceil(e);
}
function cj(e, t) {
  return e = Nx(e), de(Ur.count(Nt(e), e) + (Nt(e).getUTCDay() === 4), t, 2);
}
function lj(e) {
  return e.getUTCDay();
}
function fj(e, t) {
  return de(pa.count(Nt(e) - 1, e), t, 2);
}
function dj(e, t) {
  return de(e.getUTCFullYear() % 100, t, 2);
}
function pj(e, t) {
  return e = Nx(e), de(e.getUTCFullYear() % 100, t, 2);
}
function hj(e, t) {
  return de(e.getUTCFullYear() % 1e4, t, 4);
}
function vj(e, t) {
  var r = e.getUTCDay();
  return e = r >= 4 || r === 0 ? Ur(e) : Ur.ceil(e), de(e.getUTCFullYear() % 1e4, t, 4);
}
function yj() {
  return "+0000";
}
function Mm() {
  return "%";
}
function Im(e) {
  return +e;
}
function $m(e) {
  return Math.floor(+e / 1e3);
}
var Sr, Dx, qx;
mj({
  dateTime: "%x, %X",
  date: "%-m/%-d/%Y",
  time: "%-I:%M:%S %p",
  periods: ["AM", "PM"],
  days: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
  shortDays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
  months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
  shortMonths: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
});
function mj(e) {
  return Sr = mT(e), Dx = Sr.format, Sr.parse, qx = Sr.utcFormat, Sr.utcParse, Sr;
}
function gj(e) {
  return new Date(e);
}
function bj(e) {
  return e instanceof Date ? +e : +/* @__PURE__ */ new Date(+e);
}
function qd(e, t, r, n, i, a, o, u, s, c) {
  var f = Sd(), l = f.invert, d = f.domain, p = c(".%L"), y = c(":%S"), v = c("%I:%M"), h = c("%I %p"), g = c("%a %d"), w = c("%b %d"), b = c("%B"), O = c("%Y");
  function m(x) {
    return (s(x) < x ? p : u(x) < x ? y : o(x) < x ? v : a(x) < x ? h : n(x) < x ? i(x) < x ? g : w : r(x) < x ? b : O)(x);
  }
  return f.invert = function(x) {
    return new Date(l(x));
  }, f.domain = function(x) {
    return arguments.length ? d(Array.from(x, bj)) : d().map(gj);
  }, f.ticks = function(x) {
    var _ = d();
    return e(_[0], _[_.length - 1], x ?? 10);
  }, f.tickFormat = function(x, _) {
    return _ == null ? m : c(_);
  }, f.nice = function(x) {
    var _ = d();
    return (!x || typeof x.range != "function") && (x = t(_[0], _[_.length - 1], x ?? 10)), x ? d(Sx(_, x)) : f;
  }, f.copy = function() {
    return Oi(f, qd(e, t, r, n, i, a, o, u, s, c));
  }, f;
}
function xj() {
  return st.apply(qd(vT, yT, kt, Nd, no, _i, Rd, Id, sr, Dx).domain([new Date(2e3, 0, 1), new Date(2e3, 0, 2)]), arguments);
}
function wj() {
  return st.apply(qd(pT, hT, Nt, Dd, io, ro, kd, $d, sr, qx).domain([Date.UTC(2e3, 0, 1), Date.UTC(2e3, 0, 2)]), arguments);
}
function ao() {
  var e = 0, t = 1, r, n, i, a, o = Ke, u = !1, s;
  function c(l) {
    return l == null || isNaN(l = +l) ? s : o(i === 0 ? 0.5 : (l = (a(l) - r) * i, u ? Math.max(0, Math.min(1, l)) : l));
  }
  c.domain = function(l) {
    return arguments.length ? ([e, t] = l, r = a(e = +e), n = a(t = +t), i = r === n ? 0 : 1 / (n - r), c) : [e, t];
  }, c.clamp = function(l) {
    return arguments.length ? (u = !!l, c) : u;
  }, c.interpolator = function(l) {
    return arguments.length ? (o = l, c) : o;
  };
  function f(l) {
    return function(d) {
      var p, y;
      return arguments.length ? ([p, y] = d, o = l(p, y), c) : [o(0), o(1)];
    };
  }
  return c.range = f(ln), c.rangeRound = f(_d), c.unknown = function(l) {
    return arguments.length ? (s = l, c) : s;
  }, function(l) {
    return a = l, r = l(e), n = l(t), i = r === n ? 0 : 1 / (n - r), c;
  };
}
function er(e, t) {
  return t.domain(e.domain()).interpolator(e.interpolator()).clamp(e.clamp()).unknown(e.unknown());
}
function Lx() {
  var e = Qt(ao()(Ke));
  return e.copy = function() {
    return er(e, Lx());
  }, Ft.apply(e, arguments);
}
function Bx() {
  var e = Ed(ao()).domain([1, 10]);
  return e.copy = function() {
    return er(e, Bx()).base(e.base());
  }, Ft.apply(e, arguments);
}
function Fx() {
  var e = Td(ao());
  return e.copy = function() {
    return er(e, Fx()).constant(e.constant());
  }, Ft.apply(e, arguments);
}
function Ld() {
  var e = jd(ao());
  return e.copy = function() {
    return er(e, Ld()).exponent(e.exponent());
  }, Ft.apply(e, arguments);
}
function Oj() {
  return Ld.apply(null, arguments).exponent(0.5);
}
function zx() {
  var e = [], t = Ke;
  function r(n) {
    if (n != null && !isNaN(n = +n)) return t((xi(e, n, 1) - 1) / (e.length - 1));
  }
  return r.domain = function(n) {
    if (!arguments.length) return e.slice();
    e = [];
    for (let i of n) i != null && !isNaN(i = +i) && e.push(i);
    return e.sort(Kt), r;
  }, r.interpolator = function(n) {
    return arguments.length ? (t = n, r) : t;
  }, r.range = function() {
    return e.map((n, i) => t(i / (e.length - 1)));
  }, r.quantiles = function(n) {
    return Array.from({ length: n + 1 }, (i, a) => sE(e, a / n));
  }, r.copy = function() {
    return zx(t).domain(e);
  }, Ft.apply(r, arguments);
}
function oo() {
  var e = 0, t = 0.5, r = 1, n = 1, i, a, o, u, s, c = Ke, f, l = !1, d;
  function p(v) {
    return isNaN(v = +v) ? d : (v = 0.5 + ((v = +f(v)) - a) * (n * v < n * a ? u : s), c(l ? Math.max(0, Math.min(1, v)) : v));
  }
  p.domain = function(v) {
    return arguments.length ? ([e, t, r] = v, i = f(e = +e), a = f(t = +t), o = f(r = +r), u = i === a ? 0 : 0.5 / (a - i), s = a === o ? 0 : 0.5 / (o - a), n = a < i ? -1 : 1, p) : [e, t, r];
  }, p.clamp = function(v) {
    return arguments.length ? (l = !!v, p) : l;
  }, p.interpolator = function(v) {
    return arguments.length ? (c = v, p) : c;
  };
  function y(v) {
    return function(h) {
      var g, w, b;
      return arguments.length ? ([g, w, b] = h, c = RE(v, [g, w, b]), p) : [c(0), c(0.5), c(1)];
    };
  }
  return p.range = y(ln), p.rangeRound = y(_d), p.unknown = function(v) {
    return arguments.length ? (d = v, p) : d;
  }, function(v) {
    return f = v, i = v(e), a = v(t), o = v(r), u = i === a ? 0 : 0.5 / (a - i), s = a === o ? 0 : 0.5 / (o - a), n = a < i ? -1 : 1, p;
  };
}
function Ux() {
  var e = Qt(oo()(Ke));
  return e.copy = function() {
    return er(e, Ux());
  }, Ft.apply(e, arguments);
}
function Wx() {
  var e = Ed(oo()).domain([0.1, 1, 10]);
  return e.copy = function() {
    return er(e, Wx()).base(e.base());
  }, Ft.apply(e, arguments);
}
function Gx() {
  var e = Td(oo());
  return e.copy = function() {
    return er(e, Gx()).constant(e.constant());
  }, Ft.apply(e, arguments);
}
function Bd() {
  var e = jd(oo());
  return e.copy = function() {
    return er(e, Bd()).exponent(e.exponent());
  }, Ft.apply(e, arguments);
}
function _j() {
  return Bd.apply(null, arguments).exponent(0.5);
}
const Rm = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  scaleBand: zn,
  scaleDiverging: Ux,
  scaleDivergingLog: Wx,
  scaleDivergingPow: Bd,
  scaleDivergingSqrt: _j,
  scaleDivergingSymlog: Gx,
  scaleIdentity: _x,
  scaleImplicit: Xl,
  scaleLinear: la,
  scaleLog: Px,
  scaleOrdinal: xd,
  scalePoint: In,
  scalePow: Cd,
  scaleQuantile: Tx,
  scaleQuantize: jx,
  scaleRadial: Ex,
  scaleSequential: Lx,
  scaleSequentialLog: Bx,
  scaleSequentialPow: Ld,
  scaleSequentialQuantile: zx,
  scaleSequentialSqrt: Oj,
  scaleSequentialSymlog: Fx,
  scaleSqrt: nT,
  scaleSymlog: Ax,
  scaleThreshold: Cx,
  scaleTime: xj,
  scaleUtc: wj,
  tickFormat: Ox
}, Symbol.toStringTag, { value: "Module" }));
var Dc, km;
function Hx() {
  if (km) return Dc;
  km = 1;
  var e = un();
  function t(r, n, i) {
    for (var a = -1, o = r.length; ++a < o; ) {
      var u = r[a], s = n(u);
      if (s != null && (c === void 0 ? s === s && !e(s) : i(s, c)))
        var c = s, f = u;
    }
    return f;
  }
  return Dc = t, Dc;
}
var qc, Nm;
function Sj() {
  if (Nm) return qc;
  Nm = 1;
  function e(t, r) {
    return t > r;
  }
  return qc = e, qc;
}
var Lc, Dm;
function Pj() {
  if (Dm) return Lc;
  Dm = 1;
  var e = Hx(), t = Sj(), r = cn();
  function n(i) {
    return i && i.length ? e(i, r, t) : void 0;
  }
  return Lc = n, Lc;
}
var Aj = Pj();
const uo = /* @__PURE__ */ Pe(Aj);
var Bc, qm;
function Ej() {
  if (qm) return Bc;
  qm = 1;
  function e(t, r) {
    return t < r;
  }
  return Bc = e, Bc;
}
var Fc, Lm;
function Tj() {
  if (Lm) return Fc;
  Lm = 1;
  var e = Hx(), t = Ej(), r = cn();
  function n(i) {
    return i && i.length ? e(i, r, t) : void 0;
  }
  return Fc = n, Fc;
}
var jj = Tj();
const so = /* @__PURE__ */ Pe(jj);
var zc, Bm;
function Cj() {
  if (Bm) return zc;
  Bm = 1;
  var e = rd(), t = Jt(), r = tx(), n = Xe();
  function i(a, o) {
    var u = n(a) ? e : r;
    return u(a, t(o, 3));
  }
  return zc = i, zc;
}
var Uc, Fm;
function Mj() {
  if (Fm) return Uc;
  Fm = 1;
  var e = Q0(), t = Cj();
  function r(n, i) {
    return e(t(n, i), 1);
  }
  return Uc = r, Uc;
}
var Ij = Mj();
const $j = /* @__PURE__ */ Pe(Ij);
var Wc, zm;
function Rj() {
  if (zm) return Wc;
  zm = 1;
  var e = vd();
  function t(r, n) {
    return e(r, n);
  }
  return Wc = t, Wc;
}
var kj = Rj();
const co = /* @__PURE__ */ Pe(kj);
var fn = 1e9, Nj = {
  // These values must be integers within the stated ranges (inclusive).
  // Most of these values can be changed during run-time using `Decimal.config`.
  // The maximum number of significant digits of the result of a calculation or base conversion.
  // E.g. `Decimal.config({ precision: 20 });`
  precision: 20,
  // 1 to MAX_DIGITS
  // The rounding mode used by default by `toInteger`, `toDecimalPlaces`, `toExponential`,
  // `toFixed`, `toPrecision` and `toSignificantDigits`.
  //
  // ROUND_UP         0 Away from zero.
  // ROUND_DOWN       1 Towards zero.
  // ROUND_CEIL       2 Towards +Infinity.
  // ROUND_FLOOR      3 Towards -Infinity.
  // ROUND_HALF_UP    4 Towards nearest neighbour. If equidistant, up.
  // ROUND_HALF_DOWN  5 Towards nearest neighbour. If equidistant, down.
  // ROUND_HALF_EVEN  6 Towards nearest neighbour. If equidistant, towards even neighbour.
  // ROUND_HALF_CEIL  7 Towards nearest neighbour. If equidistant, towards +Infinity.
  // ROUND_HALF_FLOOR 8 Towards nearest neighbour. If equidistant, towards -Infinity.
  //
  // E.g.
  // `Decimal.rounding = 4;`
  // `Decimal.rounding = Decimal.ROUND_HALF_UP;`
  rounding: 4,
  // 0 to 8
  // The exponent value at and beneath which `toString` returns exponential notation.
  // JavaScript numbers: -7
  toExpNeg: -7,
  // 0 to -MAX_E
  // The exponent value at and above which `toString` returns exponential notation.
  // JavaScript numbers: 21
  toExpPos: 21,
  // 0 to MAX_E
  // The natural logarithm of 10.
  // 115 digits
  LN10: "2.302585092994045684017991454684364207601101488628772976033327900967572609677352480235997205089598298341967784042286"
}, zd, Ee = !0, ut = "[DecimalError] ", pr = ut + "Invalid argument: ", Fd = ut + "Exponent out of range: ", dn = Math.floor, or = Math.pow, Dj = /^(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i, et, qe = 1e7, Ae = 7, Kx = 9007199254740991, ha = dn(Kx / Ae), Z = {};
Z.absoluteValue = Z.abs = function() {
  var e = new this.constructor(this);
  return e.s && (e.s = 1), e;
};
Z.comparedTo = Z.cmp = function(e) {
  var t, r, n, i, a = this;
  if (e = new a.constructor(e), a.s !== e.s) return a.s || -e.s;
  if (a.e !== e.e) return a.e > e.e ^ a.s < 0 ? 1 : -1;
  for (n = a.d.length, i = e.d.length, t = 0, r = n < i ? n : i; t < r; ++t)
    if (a.d[t] !== e.d[t]) return a.d[t] > e.d[t] ^ a.s < 0 ? 1 : -1;
  return n === i ? 0 : n > i ^ a.s < 0 ? 1 : -1;
};
Z.decimalPlaces = Z.dp = function() {
  var e = this, t = e.d.length - 1, r = (t - e.e) * Ae;
  if (t = e.d[t], t) for (; t % 10 == 0; t /= 10) r--;
  return r < 0 ? 0 : r;
};
Z.dividedBy = Z.div = function(e) {
  return $t(this, new this.constructor(e));
};
Z.dividedToIntegerBy = Z.idiv = function(e) {
  var t = this, r = t.constructor;
  return we($t(t, new r(e), 0, 1), r.precision);
};
Z.equals = Z.eq = function(e) {
  return !this.cmp(e);
};
Z.exponent = function() {
  return $e(this);
};
Z.greaterThan = Z.gt = function(e) {
  return this.cmp(e) > 0;
};
Z.greaterThanOrEqualTo = Z.gte = function(e) {
  return this.cmp(e) >= 0;
};
Z.isInteger = Z.isint = function() {
  return this.e > this.d.length - 2;
};
Z.isNegative = Z.isneg = function() {
  return this.s < 0;
};
Z.isPositive = Z.ispos = function() {
  return this.s > 0;
};
Z.isZero = function() {
  return this.s === 0;
};
Z.lessThan = Z.lt = function(e) {
  return this.cmp(e) < 0;
};
Z.lessThanOrEqualTo = Z.lte = function(e) {
  return this.cmp(e) < 1;
};
Z.logarithm = Z.log = function(e) {
  var t, r = this, n = r.constructor, i = n.precision, a = i + 5;
  if (e === void 0)
    e = new n(10);
  else if (e = new n(e), e.s < 1 || e.eq(et)) throw Error(ut + "NaN");
  if (r.s < 1) throw Error(ut + (r.s ? "NaN" : "-Infinity"));
  return r.eq(et) ? new n(0) : (Ee = !1, t = $t(Kn(r, a), Kn(e, a), a), Ee = !0, we(t, i));
};
Z.minus = Z.sub = function(e) {
  var t = this;
  return e = new t.constructor(e), t.s == e.s ? Xx(t, e) : Vx(t, (e.s = -e.s, e));
};
Z.modulo = Z.mod = function(e) {
  var t, r = this, n = r.constructor, i = n.precision;
  if (e = new n(e), !e.s) throw Error(ut + "NaN");
  return r.s ? (Ee = !1, t = $t(r, e, 0, 1).times(e), Ee = !0, r.minus(t)) : we(new n(r), i);
};
Z.naturalExponential = Z.exp = function() {
  return Yx(this);
};
Z.naturalLogarithm = Z.ln = function() {
  return Kn(this);
};
Z.negated = Z.neg = function() {
  var e = new this.constructor(this);
  return e.s = -e.s || 0, e;
};
Z.plus = Z.add = function(e) {
  var t = this;
  return e = new t.constructor(e), t.s == e.s ? Vx(t, e) : Xx(t, (e.s = -e.s, e));
};
Z.precision = Z.sd = function(e) {
  var t, r, n, i = this;
  if (e !== void 0 && e !== !!e && e !== 1 && e !== 0) throw Error(pr + e);
  if (t = $e(i) + 1, n = i.d.length - 1, r = n * Ae + 1, n = i.d[n], n) {
    for (; n % 10 == 0; n /= 10) r--;
    for (n = i.d[0]; n >= 10; n /= 10) r++;
  }
  return e && t > r ? t : r;
};
Z.squareRoot = Z.sqrt = function() {
  var e, t, r, n, i, a, o, u = this, s = u.constructor;
  if (u.s < 1) {
    if (!u.s) return new s(0);
    throw Error(ut + "NaN");
  }
  for (e = $e(u), Ee = !1, i = Math.sqrt(+u), i == 0 || i == 1 / 0 ? (t = bt(u.d), (t.length + e) % 2 == 0 && (t += "0"), i = Math.sqrt(t), e = dn((e + 1) / 2) - (e < 0 || e % 2), i == 1 / 0 ? t = "5e" + e : (t = i.toExponential(), t = t.slice(0, t.indexOf("e") + 1) + e), n = new s(t)) : n = new s(i.toString()), r = s.precision, i = o = r + 3; ; )
    if (a = n, n = a.plus($t(u, a, o + 2)).times(0.5), bt(a.d).slice(0, o) === (t = bt(n.d)).slice(0, o)) {
      if (t = t.slice(o - 3, o + 1), i == o && t == "4999") {
        if (we(a, r + 1, 0), a.times(a).eq(u)) {
          n = a;
          break;
        }
      } else if (t != "9999")
        break;
      o += 4;
    }
  return Ee = !0, we(n, r);
};
Z.times = Z.mul = function(e) {
  var t, r, n, i, a, o, u, s, c, f = this, l = f.constructor, d = f.d, p = (e = new l(e)).d;
  if (!f.s || !e.s) return new l(0);
  for (e.s *= f.s, r = f.e + e.e, s = d.length, c = p.length, s < c && (a = d, d = p, p = a, o = s, s = c, c = o), a = [], o = s + c, n = o; n--; ) a.push(0);
  for (n = c; --n >= 0; ) {
    for (t = 0, i = s + n; i > n; )
      u = a[i] + p[n] * d[i - n - 1] + t, a[i--] = u % qe | 0, t = u / qe | 0;
    a[i] = (a[i] + t) % qe | 0;
  }
  for (; !a[--o]; ) a.pop();
  return t ? ++r : a.shift(), e.d = a, e.e = r, Ee ? we(e, l.precision) : e;
};
Z.toDecimalPlaces = Z.todp = function(e, t) {
  var r = this, n = r.constructor;
  return r = new n(r), e === void 0 ? r : (_t(e, 0, fn), t === void 0 ? t = n.rounding : _t(t, 0, 8), we(r, e + $e(r) + 1, t));
};
Z.toExponential = function(e, t) {
  var r, n = this, i = n.constructor;
  return e === void 0 ? r = vr(n, !0) : (_t(e, 0, fn), t === void 0 ? t = i.rounding : _t(t, 0, 8), n = we(new i(n), e + 1, t), r = vr(n, !0, e + 1)), r;
};
Z.toFixed = function(e, t) {
  var r, n, i = this, a = i.constructor;
  return e === void 0 ? vr(i) : (_t(e, 0, fn), t === void 0 ? t = a.rounding : _t(t, 0, 8), n = we(new a(i), e + $e(i) + 1, t), r = vr(n.abs(), !1, e + $e(n) + 1), i.isneg() && !i.isZero() ? "-" + r : r);
};
Z.toInteger = Z.toint = function() {
  var e = this, t = e.constructor;
  return we(new t(e), $e(e) + 1, t.rounding);
};
Z.toNumber = function() {
  return +this;
};
Z.toPower = Z.pow = function(e) {
  var t, r, n, i, a, o, u = this, s = u.constructor, c = 12, f = +(e = new s(e));
  if (!e.s) return new s(et);
  if (u = new s(u), !u.s) {
    if (e.s < 1) throw Error(ut + "Infinity");
    return u;
  }
  if (u.eq(et)) return u;
  if (n = s.precision, e.eq(et)) return we(u, n);
  if (t = e.e, r = e.d.length - 1, o = t >= r, a = u.s, o) {
    if ((r = f < 0 ? -f : f) <= Kx) {
      for (i = new s(et), t = Math.ceil(n / Ae + 4), Ee = !1; r % 2 && (i = i.times(u), Wm(i.d, t)), r = dn(r / 2), r !== 0; )
        u = u.times(u), Wm(u.d, t);
      return Ee = !0, e.s < 0 ? new s(et).div(i) : we(i, n);
    }
  } else if (a < 0) throw Error(ut + "NaN");
  return a = a < 0 && e.d[Math.max(t, r)] & 1 ? -1 : 1, u.s = 1, Ee = !1, i = e.times(Kn(u, n + c)), Ee = !0, i = Yx(i), i.s = a, i;
};
Z.toPrecision = function(e, t) {
  var r, n, i = this, a = i.constructor;
  return e === void 0 ? (r = $e(i), n = vr(i, r <= a.toExpNeg || r >= a.toExpPos)) : (_t(e, 1, fn), t === void 0 ? t = a.rounding : _t(t, 0, 8), i = we(new a(i), e, t), r = $e(i), n = vr(i, e <= r || r <= a.toExpNeg, e)), n;
};
Z.toSignificantDigits = Z.tosd = function(e, t) {
  var r = this, n = r.constructor;
  return e === void 0 ? (e = n.precision, t = n.rounding) : (_t(e, 1, fn), t === void 0 ? t = n.rounding : _t(t, 0, 8)), we(new n(r), e, t);
};
Z.toString = Z.valueOf = Z.val = Z.toJSON = Z[Symbol.for("nodejs.util.inspect.custom")] = function() {
  var e = this, t = $e(e), r = e.constructor;
  return vr(e, t <= r.toExpNeg || t >= r.toExpPos);
};
function Vx(e, t) {
  var r, n, i, a, o, u, s, c, f = e.constructor, l = f.precision;
  if (!e.s || !t.s)
    return t.s || (t = new f(e)), Ee ? we(t, l) : t;
  if (s = e.d, c = t.d, o = e.e, i = t.e, s = s.slice(), a = o - i, a) {
    for (a < 0 ? (n = s, a = -a, u = c.length) : (n = c, i = o, u = s.length), o = Math.ceil(l / Ae), u = o > u ? o + 1 : u + 1, a > u && (a = u, n.length = 1), n.reverse(); a--; ) n.push(0);
    n.reverse();
  }
  for (u = s.length, a = c.length, u - a < 0 && (a = u, n = c, c = s, s = n), r = 0; a; )
    r = (s[--a] = s[a] + c[a] + r) / qe | 0, s[a] %= qe;
  for (r && (s.unshift(r), ++i), u = s.length; s[--u] == 0; ) s.pop();
  return t.d = s, t.e = i, Ee ? we(t, l) : t;
}
function _t(e, t, r) {
  if (e !== ~~e || e < t || e > r)
    throw Error(pr + e);
}
function bt(e) {
  var t, r, n, i = e.length - 1, a = "", o = e[0];
  if (i > 0) {
    for (a += o, t = 1; t < i; t++)
      n = e[t] + "", r = Ae - n.length, r && (a += Ut(r)), a += n;
    o = e[t], n = o + "", r = Ae - n.length, r && (a += Ut(r));
  } else if (o === 0)
    return "0";
  for (; o % 10 === 0; ) o /= 10;
  return a + o;
}
var $t = /* @__PURE__ */ function() {
  function e(n, i) {
    var a, o = 0, u = n.length;
    for (n = n.slice(); u--; )
      a = n[u] * i + o, n[u] = a % qe | 0, o = a / qe | 0;
    return o && n.unshift(o), n;
  }
  function t(n, i, a, o) {
    var u, s;
    if (a != o)
      s = a > o ? 1 : -1;
    else
      for (u = s = 0; u < a; u++)
        if (n[u] != i[u]) {
          s = n[u] > i[u] ? 1 : -1;
          break;
        }
    return s;
  }
  function r(n, i, a) {
    for (var o = 0; a--; )
      n[a] -= o, o = n[a] < i[a] ? 1 : 0, n[a] = o * qe + n[a] - i[a];
    for (; !n[0] && n.length > 1; ) n.shift();
  }
  return function(n, i, a, o) {
    var u, s, c, f, l, d, p, y, v, h, g, w, b, O, m, x, _, P, E = n.constructor, I = n.s == i.s ? 1 : -1, S = n.d, j = i.d;
    if (!n.s) return new E(n);
    if (!i.s) throw Error(ut + "Division by zero");
    for (s = n.e - i.e, _ = j.length, m = S.length, p = new E(I), y = p.d = [], c = 0; j[c] == (S[c] || 0); ) ++c;
    if (j[c] > (S[c] || 0) && --s, a == null ? w = a = E.precision : o ? w = a + ($e(n) - $e(i)) + 1 : w = a, w < 0) return new E(0);
    if (w = w / Ae + 2 | 0, c = 0, _ == 1)
      for (f = 0, j = j[0], w++; (c < m || f) && w--; c++)
        b = f * qe + (S[c] || 0), y[c] = b / j | 0, f = b % j | 0;
    else {
      for (f = qe / (j[0] + 1) | 0, f > 1 && (j = e(j, f), S = e(S, f), _ = j.length, m = S.length), O = _, v = S.slice(0, _), h = v.length; h < _; ) v[h++] = 0;
      P = j.slice(), P.unshift(0), x = j[0], j[1] >= qe / 2 && ++x;
      do
        f = 0, u = t(j, v, _, h), u < 0 ? (g = v[0], _ != h && (g = g * qe + (v[1] || 0)), f = g / x | 0, f > 1 ? (f >= qe && (f = qe - 1), l = e(j, f), d = l.length, h = v.length, u = t(l, v, d, h), u == 1 && (f--, r(l, _ < d ? P : j, d))) : (f == 0 && (u = f = 1), l = j.slice()), d = l.length, d < h && l.unshift(0), r(v, l, h), u == -1 && (h = v.length, u = t(j, v, _, h), u < 1 && (f++, r(v, _ < h ? P : j, h))), h = v.length) : u === 0 && (f++, v = [0]), y[c++] = f, u && v[0] ? v[h++] = S[O] || 0 : (v = [S[O]], h = 1);
      while ((O++ < m || v[0] !== void 0) && w--);
    }
    return y[0] || y.shift(), p.e = s, we(p, o ? a + $e(p) + 1 : a);
  };
}();
function Yx(e, t) {
  var r, n, i, a, o, u, s = 0, c = 0, f = e.constructor, l = f.precision;
  if ($e(e) > 16) throw Error(Fd + $e(e));
  if (!e.s) return new f(et);
  for (t == null ? (Ee = !1, u = l) : u = t, o = new f(0.03125); e.abs().gte(0.1); )
    e = e.times(o), c += 5;
  for (n = Math.log(or(2, c)) / Math.LN10 * 2 + 5 | 0, u += n, r = i = a = new f(et), f.precision = u; ; ) {
    if (i = we(i.times(e), u), r = r.times(++s), o = a.plus($t(i, r, u)), bt(o.d).slice(0, u) === bt(a.d).slice(0, u)) {
      for (; c--; ) a = we(a.times(a), u);
      return f.precision = l, t == null ? (Ee = !0, we(a, l)) : a;
    }
    a = o;
  }
}
function $e(e) {
  for (var t = e.e * Ae, r = e.d[0]; r >= 10; r /= 10) t++;
  return t;
}
function Gc(e, t, r) {
  if (t > e.LN10.sd())
    throw Ee = !0, r && (e.precision = r), Error(ut + "LN10 precision limit exceeded");
  return we(new e(e.LN10), t);
}
function Ut(e) {
  for (var t = ""; e--; ) t += "0";
  return t;
}
function Kn(e, t) {
  var r, n, i, a, o, u, s, c, f, l = 1, d = 10, p = e, y = p.d, v = p.constructor, h = v.precision;
  if (p.s < 1) throw Error(ut + (p.s ? "NaN" : "-Infinity"));
  if (p.eq(et)) return new v(0);
  if (t == null ? (Ee = !1, c = h) : c = t, p.eq(10))
    return t == null && (Ee = !0), Gc(v, c);
  if (c += d, v.precision = c, r = bt(y), n = r.charAt(0), a = $e(p), Math.abs(a) < 15e14) {
    for (; n < 7 && n != 1 || n == 1 && r.charAt(1) > 3; )
      p = p.times(e), r = bt(p.d), n = r.charAt(0), l++;
    a = $e(p), n > 1 ? (p = new v("0." + r), a++) : p = new v(n + "." + r.slice(1));
  } else
    return s = Gc(v, c + 2, h).times(a + ""), p = Kn(new v(n + "." + r.slice(1)), c - d).plus(s), v.precision = h, t == null ? (Ee = !0, we(p, h)) : p;
  for (u = o = p = $t(p.minus(et), p.plus(et), c), f = we(p.times(p), c), i = 3; ; ) {
    if (o = we(o.times(f), c), s = u.plus($t(o, new v(i), c)), bt(s.d).slice(0, c) === bt(u.d).slice(0, c))
      return u = u.times(2), a !== 0 && (u = u.plus(Gc(v, c + 2, h).times(a + ""))), u = $t(u, new v(l), c), v.precision = h, t == null ? (Ee = !0, we(u, h)) : u;
    u = s, i += 2;
  }
}
function Um(e, t) {
  var r, n, i;
  for ((r = t.indexOf(".")) > -1 && (t = t.replace(".", "")), (n = t.search(/e/i)) > 0 ? (r < 0 && (r = n), r += +t.slice(n + 1), t = t.substring(0, n)) : r < 0 && (r = t.length), n = 0; t.charCodeAt(n) === 48; ) ++n;
  for (i = t.length; t.charCodeAt(i - 1) === 48; ) --i;
  if (t = t.slice(n, i), t) {
    if (i -= n, r = r - n - 1, e.e = dn(r / Ae), e.d = [], n = (r + 1) % Ae, r < 0 && (n += Ae), n < i) {
      for (n && e.d.push(+t.slice(0, n)), i -= Ae; n < i; ) e.d.push(+t.slice(n, n += Ae));
      t = t.slice(n), n = Ae - t.length;
    } else
      n -= i;
    for (; n--; ) t += "0";
    if (e.d.push(+t), Ee && (e.e > ha || e.e < -ha)) throw Error(Fd + r);
  } else
    e.s = 0, e.e = 0, e.d = [0];
  return e;
}
function we(e, t, r) {
  var n, i, a, o, u, s, c, f, l = e.d;
  for (o = 1, a = l[0]; a >= 10; a /= 10) o++;
  if (n = t - o, n < 0)
    n += Ae, i = t, c = l[f = 0];
  else {
    if (f = Math.ceil((n + 1) / Ae), a = l.length, f >= a) return e;
    for (c = a = l[f], o = 1; a >= 10; a /= 10) o++;
    n %= Ae, i = n - Ae + o;
  }
  if (r !== void 0 && (a = or(10, o - i - 1), u = c / a % 10 | 0, s = t < 0 || l[f + 1] !== void 0 || c % a, s = r < 4 ? (u || s) && (r == 0 || r == (e.s < 0 ? 3 : 2)) : u > 5 || u == 5 && (r == 4 || s || r == 6 && // Check whether the digit to the left of the rounding digit is odd.
  (n > 0 ? i > 0 ? c / or(10, o - i) : 0 : l[f - 1]) % 10 & 1 || r == (e.s < 0 ? 8 : 7))), t < 1 || !l[0])
    return s ? (a = $e(e), l.length = 1, t = t - a - 1, l[0] = or(10, (Ae - t % Ae) % Ae), e.e = dn(-t / Ae) || 0) : (l.length = 1, l[0] = e.e = e.s = 0), e;
  if (n == 0 ? (l.length = f, a = 1, f--) : (l.length = f + 1, a = or(10, Ae - n), l[f] = i > 0 ? (c / or(10, o - i) % or(10, i) | 0) * a : 0), s)
    for (; ; )
      if (f == 0) {
        (l[0] += a) == qe && (l[0] = 1, ++e.e);
        break;
      } else {
        if (l[f] += a, l[f] != qe) break;
        l[f--] = 0, a = 1;
      }
  for (n = l.length; l[--n] === 0; ) l.pop();
  if (Ee && (e.e > ha || e.e < -ha))
    throw Error(Fd + $e(e));
  return e;
}
function Xx(e, t) {
  var r, n, i, a, o, u, s, c, f, l, d = e.constructor, p = d.precision;
  if (!e.s || !t.s)
    return t.s ? t.s = -t.s : t = new d(e), Ee ? we(t, p) : t;
  if (s = e.d, l = t.d, n = t.e, c = e.e, s = s.slice(), o = c - n, o) {
    for (f = o < 0, f ? (r = s, o = -o, u = l.length) : (r = l, n = c, u = s.length), i = Math.max(Math.ceil(p / Ae), u) + 2, o > i && (o = i, r.length = 1), r.reverse(), i = o; i--; ) r.push(0);
    r.reverse();
  } else {
    for (i = s.length, u = l.length, f = i < u, f && (u = i), i = 0; i < u; i++)
      if (s[i] != l[i]) {
        f = s[i] < l[i];
        break;
      }
    o = 0;
  }
  for (f && (r = s, s = l, l = r, t.s = -t.s), u = s.length, i = l.length - u; i > 0; --i) s[u++] = 0;
  for (i = l.length; i > o; ) {
    if (s[--i] < l[i]) {
      for (a = i; a && s[--a] === 0; ) s[a] = qe - 1;
      --s[a], s[i] += qe;
    }
    s[i] -= l[i];
  }
  for (; s[--u] === 0; ) s.pop();
  for (; s[0] === 0; s.shift()) --n;
  return s[0] ? (t.d = s, t.e = n, Ee ? we(t, p) : t) : new d(0);
}
function vr(e, t, r) {
  var n, i = $e(e), a = bt(e.d), o = a.length;
  return t ? (r && (n = r - o) > 0 ? a = a.charAt(0) + "." + a.slice(1) + Ut(n) : o > 1 && (a = a.charAt(0) + "." + a.slice(1)), a = a + (i < 0 ? "e" : "e+") + i) : i < 0 ? (a = "0." + Ut(-i - 1) + a, r && (n = r - o) > 0 && (a += Ut(n))) : i >= o ? (a += Ut(i + 1 - o), r && (n = r - i - 1) > 0 && (a = a + "." + Ut(n))) : ((n = i + 1) < o && (a = a.slice(0, n) + "." + a.slice(n)), r && (n = r - o) > 0 && (i + 1 === o && (a += "."), a += Ut(n))), e.s < 0 ? "-" + a : a;
}
function Wm(e, t) {
  if (e.length > t)
    return e.length = t, !0;
}
function Zx(e) {
  var t, r, n;
  function i(a) {
    var o = this;
    if (!(o instanceof i)) return new i(a);
    if (o.constructor = i, a instanceof i) {
      o.s = a.s, o.e = a.e, o.d = (a = a.d) ? a.slice() : a;
      return;
    }
    if (typeof a == "number") {
      if (a * 0 !== 0)
        throw Error(pr + a);
      if (a > 0)
        o.s = 1;
      else if (a < 0)
        a = -a, o.s = -1;
      else {
        o.s = 0, o.e = 0, o.d = [0];
        return;
      }
      if (a === ~~a && a < 1e7) {
        o.e = 0, o.d = [a];
        return;
      }
      return Um(o, a.toString());
    } else if (typeof a != "string")
      throw Error(pr + a);
    if (a.charCodeAt(0) === 45 ? (a = a.slice(1), o.s = -1) : o.s = 1, Dj.test(a)) Um(o, a);
    else throw Error(pr + a);
  }
  if (i.prototype = Z, i.ROUND_UP = 0, i.ROUND_DOWN = 1, i.ROUND_CEIL = 2, i.ROUND_FLOOR = 3, i.ROUND_HALF_UP = 4, i.ROUND_HALF_DOWN = 5, i.ROUND_HALF_EVEN = 6, i.ROUND_HALF_CEIL = 7, i.ROUND_HALF_FLOOR = 8, i.clone = Zx, i.config = i.set = qj, e === void 0 && (e = {}), e)
    for (n = ["precision", "rounding", "toExpNeg", "toExpPos", "LN10"], t = 0; t < n.length; ) e.hasOwnProperty(r = n[t++]) || (e[r] = this[r]);
  return i.config(e), i;
}
function qj(e) {
  if (!e || typeof e != "object")
    throw Error(ut + "Object expected");
  var t, r, n, i = [
    "precision",
    1,
    fn,
    "rounding",
    0,
    8,
    "toExpNeg",
    -1 / 0,
    0,
    "toExpPos",
    0,
    1 / 0
  ];
  for (t = 0; t < i.length; t += 3)
    if ((n = e[r = i[t]]) !== void 0)
      if (dn(n) === n && n >= i[t + 1] && n <= i[t + 2]) this[r] = n;
      else throw Error(pr + r + ": " + n);
  if ((n = e[r = "LN10"]) !== void 0)
    if (n == Math.LN10) this[r] = new this(n);
    else throw Error(pr + r + ": " + n);
  return this;
}
var zd = Zx(Nj);
et = new zd(1);
const xe = zd;
function Lj(e) {
  return Uj(e) || zj(e) || Fj(e) || Bj();
}
function Bj() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Fj(e, t) {
  if (e) {
    if (typeof e == "string") return ef(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return ef(e, t);
  }
}
function zj(e) {
  if (typeof Symbol < "u" && Symbol.iterator in Object(e)) return Array.from(e);
}
function Uj(e) {
  if (Array.isArray(e)) return ef(e);
}
function ef(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++)
    n[r] = e[r];
  return n;
}
var Wj = function(t) {
  return t;
}, Jx = {
  "@@functional/placeholder": !0
}, Qx = function(t) {
  return t === Jx;
}, Gm = function(t) {
  return function r() {
    return arguments.length === 0 || arguments.length === 1 && Qx(arguments.length <= 0 ? void 0 : arguments[0]) ? r : t.apply(void 0, arguments);
  };
}, Gj = function e(t, r) {
  return t === 1 ? r : Gm(function() {
    for (var n = arguments.length, i = new Array(n), a = 0; a < n; a++)
      i[a] = arguments[a];
    var o = i.filter(function(u) {
      return u !== Jx;
    }).length;
    return o >= t ? r.apply(void 0, i) : e(t - o, Gm(function() {
      for (var u = arguments.length, s = new Array(u), c = 0; c < u; c++)
        s[c] = arguments[c];
      var f = i.map(function(l) {
        return Qx(l) ? s.shift() : l;
      });
      return r.apply(void 0, Lj(f).concat(s));
    }));
  });
}, lo = function(t) {
  return Gj(t.length, t);
}, tf = function(t, r) {
  for (var n = [], i = t; i < r; ++i)
    n[i - t] = i;
  return n;
}, Hj = lo(function(e, t) {
  return Array.isArray(t) ? t.map(e) : Object.keys(t).map(function(r) {
    return t[r];
  }).map(e);
}), Kj = function() {
  for (var t = arguments.length, r = new Array(t), n = 0; n < t; n++)
    r[n] = arguments[n];
  if (!r.length)
    return Wj;
  var i = r.reverse(), a = i[0], o = i.slice(1);
  return function() {
    return o.reduce(function(u, s) {
      return s(u);
    }, a.apply(void 0, arguments));
  };
}, rf = function(t) {
  return Array.isArray(t) ? t.reverse() : t.split("").reverse.join("");
}, ew = function(t) {
  var r = null, n = null;
  return function() {
    for (var i = arguments.length, a = new Array(i), o = 0; o < i; o++)
      a[o] = arguments[o];
    return r && a.every(function(u, s) {
      return u === r[s];
    }) || (r = a, n = t.apply(void 0, a)), n;
  };
};
function Vj(e) {
  var t;
  return e === 0 ? t = 1 : t = Math.floor(new xe(e).abs().log(10).toNumber()) + 1, t;
}
function Yj(e, t, r) {
  for (var n = new xe(e), i = 0, a = []; n.lt(t) && i < 1e5; )
    a.push(n.toNumber()), n = n.add(r), i++;
  return a;
}
var Xj = lo(function(e, t, r) {
  var n = +e, i = +t;
  return n + r * (i - n);
}), Zj = lo(function(e, t, r) {
  var n = t - +e;
  return n = n || 1 / 0, (r - e) / n;
}), Jj = lo(function(e, t, r) {
  var n = t - +e;
  return n = n || 1 / 0, Math.max(0, Math.min(1, (r - e) / n));
});
const fo = {
  rangeStep: Yj,
  getDigitCount: Vj,
  interpolateNumber: Xj,
  uninterpolateNumber: Zj,
  uninterpolateTruncation: Jj
};
function nf(e) {
  return tC(e) || eC(e) || tw(e) || Qj();
}
function Qj() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function eC(e) {
  if (typeof Symbol < "u" && Symbol.iterator in Object(e)) return Array.from(e);
}
function tC(e) {
  if (Array.isArray(e)) return af(e);
}
function Vn(e, t) {
  return iC(e) || nC(e, t) || tw(e, t) || rC();
}
function rC() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function tw(e, t) {
  if (e) {
    if (typeof e == "string") return af(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return af(e, t);
  }
}
function af(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++)
    n[r] = e[r];
  return n;
}
function nC(e, t) {
  if (!(typeof Symbol > "u" || !(Symbol.iterator in Object(e)))) {
    var r = [], n = !0, i = !1, a = void 0;
    try {
      for (var o = e[Symbol.iterator](), u; !(n = (u = o.next()).done) && (r.push(u.value), !(t && r.length === t)); n = !0)
        ;
    } catch (s) {
      i = !0, a = s;
    } finally {
      try {
        !n && o.return != null && o.return();
      } finally {
        if (i) throw a;
      }
    }
    return r;
  }
}
function iC(e) {
  if (Array.isArray(e)) return e;
}
function rw(e) {
  var t = Vn(e, 2), r = t[0], n = t[1], i = r, a = n;
  return r > n && (i = n, a = r), [i, a];
}
function nw(e, t, r) {
  if (e.lte(0))
    return new xe(0);
  var n = fo.getDigitCount(e.toNumber()), i = new xe(10).pow(n), a = e.div(i), o = n !== 1 ? 0.05 : 0.1, u = new xe(Math.ceil(a.div(o).toNumber())).add(r).mul(o), s = u.mul(i);
  return t ? s : new xe(Math.ceil(s));
}
function aC(e, t, r) {
  var n = 1, i = new xe(e);
  if (!i.isint() && r) {
    var a = Math.abs(e);
    a < 1 ? (n = new xe(10).pow(fo.getDigitCount(e) - 1), i = new xe(Math.floor(i.div(n).toNumber())).mul(n)) : a > 1 && (i = new xe(Math.floor(e)));
  } else e === 0 ? i = new xe(Math.floor((t - 1) / 2)) : r || (i = new xe(Math.floor(e)));
  var o = Math.floor((t - 1) / 2), u = Kj(Hj(function(s) {
    return i.add(new xe(s - o).mul(n)).toNumber();
  }), tf);
  return u(0, t);
}
function iw(e, t, r, n) {
  var i = arguments.length > 4 && arguments[4] !== void 0 ? arguments[4] : 0;
  if (!Number.isFinite((t - e) / (r - 1)))
    return {
      step: new xe(0),
      tickMin: new xe(0),
      tickMax: new xe(0)
    };
  var a = nw(new xe(t).sub(e).div(r - 1), n, i), o;
  e <= 0 && t >= 0 ? o = new xe(0) : (o = new xe(e).add(t).div(2), o = o.sub(new xe(o).mod(a)));
  var u = Math.ceil(o.sub(e).div(a).toNumber()), s = Math.ceil(new xe(t).sub(o).div(a).toNumber()), c = u + s + 1;
  return c > r ? iw(e, t, r, n, i + 1) : (c < r && (s = t > 0 ? s + (r - c) : s, u = t > 0 ? u : u + (r - c)), {
    step: a,
    tickMin: o.sub(new xe(u).mul(a)),
    tickMax: o.add(new xe(s).mul(a))
  });
}
function oC(e) {
  var t = Vn(e, 2), r = t[0], n = t[1], i = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 6, a = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : !0, o = Math.max(i, 2), u = rw([r, n]), s = Vn(u, 2), c = s[0], f = s[1];
  if (c === -1 / 0 || f === 1 / 0) {
    var l = f === 1 / 0 ? [c].concat(nf(tf(0, i - 1).map(function() {
      return 1 / 0;
    }))) : [].concat(nf(tf(0, i - 1).map(function() {
      return -1 / 0;
    })), [f]);
    return r > n ? rf(l) : l;
  }
  if (c === f)
    return aC(c, i, a);
  var d = iw(c, f, o, a), p = d.step, y = d.tickMin, v = d.tickMax, h = fo.rangeStep(y, v.add(new xe(0.1).mul(p)), p);
  return r > n ? rf(h) : h;
}
function uC(e, t) {
  var r = Vn(e, 2), n = r[0], i = r[1], a = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : !0, o = rw([n, i]), u = Vn(o, 2), s = u[0], c = u[1];
  if (s === -1 / 0 || c === 1 / 0)
    return [n, i];
  if (s === c)
    return [s];
  var f = Math.max(t, 2), l = nw(new xe(c).sub(s).div(f - 1), a, 0), d = [].concat(nf(fo.rangeStep(new xe(s), new xe(c).sub(new xe(0.99).mul(l)), l)), [c]);
  return n > i ? rf(d) : d;
}
var sC = ew(oC), cC = ew(uC), lC = process.env.NODE_ENV === "production", Hc = "Invariant failed";
function Ye(e, t) {
  if (lC)
    throw new Error(Hc);
  var r = typeof t == "function" ? t() : t, n = r ? "".concat(Hc, ": ").concat(r) : Hc;
  throw new Error(n);
}
var fC = ["offset", "layout", "width", "dataKey", "data", "dataPointFormatter", "xAxis", "yAxis"];
function Wr(e) {
  "@babel/helpers - typeof";
  return Wr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Wr(e);
}
function va() {
  return va = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, va.apply(this, arguments);
}
function dC(e, t) {
  return yC(e) || vC(e, t) || hC(e, t) || pC();
}
function pC() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function hC(e, t) {
  if (e) {
    if (typeof e == "string") return Hm(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Hm(e, t);
  }
}
function Hm(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function vC(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function yC(e) {
  if (Array.isArray(e)) return e;
}
function mC(e, t) {
  if (e == null) return {};
  var r = gC(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function gC(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function bC(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function xC(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, uw(n.key), n);
  }
}
function wC(e, t, r) {
  return xC(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function OC(e, t, r) {
  return t = ya(t), _C(e, aw() ? Reflect.construct(t, r || [], ya(e).constructor) : t.apply(e, r));
}
function _C(e, t) {
  if (t && (Wr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return SC(e);
}
function SC(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function aw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (aw = function() {
    return !!e;
  })();
}
function ya(e) {
  return ya = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, ya(e);
}
function PC(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && of(e, t);
}
function of(e, t) {
  return of = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, of(e, t);
}
function ow(e, t, r) {
  return t = uw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function uw(e) {
  var t = AC(e, "string");
  return Wr(t) == "symbol" ? t : t + "";
}
function AC(e, t) {
  if (Wr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Wr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Si = /* @__PURE__ */ function(e) {
  function t() {
    return bC(this, t), OC(this, t, arguments);
  }
  return PC(t, e), wC(t, [{
    key: "render",
    value: function() {
      var n = this.props, i = n.offset, a = n.layout, o = n.width, u = n.dataKey, s = n.data, c = n.dataPointFormatter, f = n.xAxis, l = n.yAxis, d = mC(n, fC), p = fe(d, !1);
      this.props.direction === "x" && f.type !== "number" && (process.env.NODE_ENV !== "production" ? Ye(!1, 'ErrorBar requires Axis type property to be "number".') : Ye());
      var y = s.map(function(v) {
        var h = c(v, u), g = h.x, w = h.y, b = h.value, O = h.errorVal;
        if (!O)
          return null;
        var m = [], x, _;
        if (Array.isArray(O)) {
          var P = dC(O, 2);
          x = P[0], _ = P[1];
        } else
          x = _ = O;
        if (a === "vertical") {
          var E = f.scale, I = w + i, S = I + o, j = I - o, M = E(b - x), R = E(b + _);
          m.push({
            x1: R,
            y1: S,
            x2: R,
            y2: j
          }), m.push({
            x1: M,
            y1: I,
            x2: R,
            y2: I
          }), m.push({
            x1: M,
            y1: S,
            x2: M,
            y2: j
          });
        } else if (a === "horizontal") {
          var k = l.scale, q = g + i, L = q - o, U = q + o, z = k(b - x), $ = k(b + _);
          m.push({
            x1: L,
            y1: $,
            x2: U,
            y2: $
          }), m.push({
            x1: q,
            y1: z,
            x2: q,
            y2: $
          }), m.push({
            x1: L,
            y1: z,
            x2: U,
            y2: z
          });
        }
        return /* @__PURE__ */ T.createElement(je, va({
          className: "recharts-errorBar",
          key: "bar-".concat(m.map(function(D) {
            return "".concat(D.x1, "-").concat(D.x2, "-").concat(D.y1, "-").concat(D.y2);
          }))
        }, p), m.map(function(D) {
          return /* @__PURE__ */ T.createElement("line", va({}, D, {
            key: "line-".concat(D.x1, "-").concat(D.x2, "-").concat(D.y1, "-").concat(D.y2)
          }));
        }));
      });
      return /* @__PURE__ */ T.createElement(je, {
        className: "recharts-errorBars"
      }, y);
    }
  }]);
}(T.Component);
ow(Si, "defaultProps", {
  stroke: "black",
  strokeWidth: 1.5,
  width: 5,
  offset: 0,
  layout: "horizontal"
});
ow(Si, "displayName", "ErrorBar");
function Yn(e) {
  "@babel/helpers - typeof";
  return Yn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Yn(e);
}
function Km(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function ir(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Km(Object(r), !0).forEach(function(n) {
      EC(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Km(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function EC(e, t, r) {
  return t = TC(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function TC(e) {
  var t = jC(e, "string");
  return Yn(t) == "symbol" ? t : t + "";
}
function jC(e, t) {
  if (Yn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Yn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var sw = function(t) {
  var r = t.children, n = t.formattedGraphicalItems, i = t.legendWidth, a = t.legendContent, o = Qe(r, Ir);
  if (!o)
    return null;
  var u = Ir.defaultProps, s = u !== void 0 ? ir(ir({}, u), o.props) : {}, c;
  return o.props && o.props.payload ? c = o.props && o.props.payload : a === "children" ? c = (n || []).reduce(function(f, l) {
    var d = l.item, p = l.props, y = p.sectors || p.data || [];
    return f.concat(y.map(function(v) {
      return {
        type: o.props.iconType || d.props.legendType,
        value: v.name,
        color: v.fill,
        payload: v
      };
    }));
  }, []) : c = (n || []).map(function(f) {
    var l = f.item, d = l.type.defaultProps, p = d !== void 0 ? ir(ir({}, d), l.props) : {}, y = p.dataKey, v = p.name, h = p.legendType, g = p.hide;
    return {
      inactive: g,
      dataKey: y,
      type: s.iconType || h || "square",
      color: Ud(l),
      value: v || y,
      // @ts-expect-error property strokeDasharray is required in Payload but optional in props
      payload: p
    };
  }), ir(ir(ir({}, s), Ir.getWithHeight(o, i)), {}, {
    payload: c,
    item: o
  });
};
function Xn(e) {
  "@babel/helpers - typeof";
  return Xn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Xn(e);
}
function Vm(e) {
  return $C(e) || IC(e) || MC(e) || CC();
}
function CC() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function MC(e, t) {
  if (e) {
    if (typeof e == "string") return uf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return uf(e, t);
  }
}
function IC(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function $C(e) {
  if (Array.isArray(e)) return uf(e);
}
function uf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function Ym(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Te(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ym(Object(r), !0).forEach(function(n) {
      Rr(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ym(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Rr(e, t, r) {
  return t = RC(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function RC(e) {
  var t = kC(e, "string");
  return Xn(t) == "symbol" ? t : t + "";
}
function kC(e, t) {
  if (Xn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Xn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function tt(e, t, r) {
  return ce(e) || ce(t) ? r : ke(t) ? at(e, t, r) : ue(t) ? t(e) : r;
}
function $n(e, t, r, n) {
  var i = $j(e, function(u) {
    return tt(u, t);
  });
  if (r === "number") {
    var a = i.filter(function(u) {
      return K(u) || parseFloat(u);
    });
    return a.length ? [so(a), uo(a)] : [1 / 0, -1 / 0];
  }
  var o = n ? i.filter(function(u) {
    return !ce(u);
  }) : i;
  return o.map(function(u) {
    return ke(u) || u instanceof Date ? u : "";
  });
}
var NC = function(t) {
  var r, n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : [], i = arguments.length > 2 ? arguments[2] : void 0, a = arguments.length > 3 ? arguments[3] : void 0, o = -1, u = (r = n == null ? void 0 : n.length) !== null && r !== void 0 ? r : 0;
  if (u <= 1)
    return 0;
  if (a && a.axisType === "angleAxis" && Math.abs(Math.abs(a.range[1] - a.range[0]) - 360) <= 1e-6)
    for (var s = a.range, c = 0; c < u; c++) {
      var f = c > 0 ? i[c - 1].coordinate : i[u - 1].coordinate, l = i[c].coordinate, d = c >= u - 1 ? i[0].coordinate : i[c + 1].coordinate, p = void 0;
      if (yt(l - f) !== yt(d - l)) {
        var y = [];
        if (yt(d - l) === yt(s[1] - s[0])) {
          p = d;
          var v = l + s[1] - s[0];
          y[0] = Math.min(v, (v + f) / 2), y[1] = Math.max(v, (v + f) / 2);
        } else {
          p = f;
          var h = d + s[1] - s[0];
          y[0] = Math.min(l, (h + l) / 2), y[1] = Math.max(l, (h + l) / 2);
        }
        var g = [Math.min(l, (p + l) / 2), Math.max(l, (p + l) / 2)];
        if (t > g[0] && t <= g[1] || t >= y[0] && t <= y[1]) {
          o = i[c].index;
          break;
        }
      } else {
        var w = Math.min(f, d), b = Math.max(f, d);
        if (t > (w + l) / 2 && t <= (b + l) / 2) {
          o = i[c].index;
          break;
        }
      }
    }
  else
    for (var O = 0; O < u; O++)
      if (O === 0 && t <= (n[O].coordinate + n[O + 1].coordinate) / 2 || O > 0 && O < u - 1 && t > (n[O].coordinate + n[O - 1].coordinate) / 2 && t <= (n[O].coordinate + n[O + 1].coordinate) / 2 || O === u - 1 && t > (n[O].coordinate + n[O - 1].coordinate) / 2) {
        o = n[O].index;
        break;
      }
  return o;
}, Ud = function(t) {
  var r, n = t, i = n.type.displayName, a = (r = t.type) !== null && r !== void 0 && r.defaultProps ? Te(Te({}, t.type.defaultProps), t.props) : t.props, o = a.stroke, u = a.fill, s;
  switch (i) {
    case "Line":
      s = o;
      break;
    case "Area":
    case "Radar":
      s = o && o !== "none" ? o : u;
      break;
    default:
      s = u;
      break;
  }
  return s;
}, DC = function(t) {
  var r = t.barSize, n = t.totalSize, i = t.stackGroups, a = i === void 0 ? {} : i;
  if (!a)
    return {};
  for (var o = {}, u = Object.keys(a), s = 0, c = u.length; s < c; s++)
    for (var f = a[u[s]].stackGroups, l = Object.keys(f), d = 0, p = l.length; d < p; d++) {
      var y = f[l[d]], v = y.items, h = y.cateAxisId, g = v.filter(function(_) {
        return Ht(_.type).indexOf("Bar") >= 0;
      });
      if (g && g.length) {
        var w = g[0].type.defaultProps, b = w !== void 0 ? Te(Te({}, w), g[0].props) : g[0].props, O = b.barSize, m = b[h];
        o[m] || (o[m] = []);
        var x = ce(O) ? r : O;
        o[m].push({
          item: g[0],
          stackList: g.slice(1),
          barSize: ce(x) ? void 0 : hr(x, n, 0)
        });
      }
    }
  return o;
}, qC = function(t) {
  var r = t.barGap, n = t.barCategoryGap, i = t.bandSize, a = t.sizeList, o = a === void 0 ? [] : a, u = t.maxBarSize, s = o.length;
  if (s < 1) return null;
  var c = hr(r, i, 0, !0), f, l = [];
  if (o[0].barSize === +o[0].barSize) {
    var d = !1, p = i / s, y = o.reduce(function(O, m) {
      return O + m.barSize || 0;
    }, 0);
    y += (s - 1) * c, y >= i && (y -= (s - 1) * c, c = 0), y >= i && p > 0 && (d = !0, p *= 0.9, y = s * p);
    var v = (i - y) / 2 >> 0, h = {
      offset: v - c,
      size: 0
    };
    f = o.reduce(function(O, m) {
      var x = {
        item: m.item,
        position: {
          offset: h.offset + h.size + c,
          // @ts-expect-error the type check above does not check for type number explicitly
          size: d ? p : m.barSize
        }
      }, _ = [].concat(Vm(O), [x]);
      return h = _[_.length - 1].position, m.stackList && m.stackList.length && m.stackList.forEach(function(P) {
        _.push({
          item: P,
          position: h
        });
      }), _;
    }, l);
  } else {
    var g = hr(n, i, 0, !0);
    i - 2 * g - (s - 1) * c <= 0 && (c = 0);
    var w = (i - 2 * g - (s - 1) * c) / s;
    w > 1 && (w >>= 0);
    var b = u === +u ? Math.min(w, u) : w;
    f = o.reduce(function(O, m, x) {
      var _ = [].concat(Vm(O), [{
        item: m.item,
        position: {
          offset: g + (w + c) * x + (w - b) / 2,
          size: b
        }
      }]);
      return m.stackList && m.stackList.length && m.stackList.forEach(function(P) {
        _.push({
          item: P,
          position: _[_.length - 1].position
        });
      }), _;
    }, l);
  }
  return f;
}, LC = function(t, r, n, i) {
  var a = n.children, o = n.width, u = n.margin, s = o - (u.left || 0) - (u.right || 0), c = sw({
    children: a,
    legendWidth: s
  });
  if (c) {
    var f = i || {}, l = f.width, d = f.height, p = c.align, y = c.verticalAlign, v = c.layout;
    if ((v === "vertical" || v === "horizontal" && y === "middle") && p !== "center" && K(t[p]))
      return Te(Te({}, t), {}, Rr({}, p, t[p] + (l || 0)));
    if ((v === "horizontal" || v === "vertical" && p === "center") && y !== "middle" && K(t[y]))
      return Te(Te({}, t), {}, Rr({}, y, t[y] + (d || 0)));
  }
  return t;
}, BC = function(t, r, n) {
  return ce(r) ? !0 : t === "horizontal" ? r === "yAxis" : t === "vertical" || n === "x" ? r === "xAxis" : n === "y" ? r === "yAxis" : !0;
}, cw = function(t, r, n, i, a) {
  var o = r.props.children, u = ot(o, Si).filter(function(c) {
    return BC(i, a, c.props.direction);
  });
  if (u && u.length) {
    var s = u.map(function(c) {
      return c.props.dataKey;
    });
    return t.reduce(function(c, f) {
      var l = tt(f, n);
      if (ce(l)) return c;
      var d = Array.isArray(l) ? [so(l), uo(l)] : [l, l], p = s.reduce(function(y, v) {
        var h = tt(f, v, 0), g = d[0] - Math.abs(Array.isArray(h) ? h[0] : h), w = d[1] + Math.abs(Array.isArray(h) ? h[1] : h);
        return [Math.min(g, y[0]), Math.max(w, y[1])];
      }, [1 / 0, -1 / 0]);
      return [Math.min(p[0], c[0]), Math.max(p[1], c[1])];
    }, [1 / 0, -1 / 0]);
  }
  return null;
}, FC = function(t, r, n, i, a) {
  var o = r.map(function(u) {
    return cw(t, u, n, a, i);
  }).filter(function(u) {
    return !ce(u);
  });
  return o && o.length ? o.reduce(function(u, s) {
    return [Math.min(u[0], s[0]), Math.max(u[1], s[1])];
  }, [1 / 0, -1 / 0]) : null;
}, lw = function(t, r, n, i, a) {
  var o = r.map(function(s) {
    var c = s.props.dataKey;
    return n === "number" && c && cw(t, s, c, i) || $n(t, c, n, a);
  });
  if (n === "number")
    return o.reduce(
      // @ts-expect-error if (type === number) means that the domain is numerical type
      // - but this link is missing in the type definition
      function(s, c) {
        return [Math.min(s[0], c[0]), Math.max(s[1], c[1])];
      },
      [1 / 0, -1 / 0]
    );
  var u = {};
  return o.reduce(function(s, c) {
    for (var f = 0, l = c.length; f < l; f++)
      u[c[f]] || (u[c[f]] = !0, s.push(c[f]));
    return s;
  }, []);
}, fw = function(t, r) {
  return t === "horizontal" && r === "xAxis" || t === "vertical" && r === "yAxis" || t === "centric" && r === "angleAxis" || t === "radial" && r === "radiusAxis";
}, dw = function(t, r, n, i) {
  if (i)
    return t.map(function(s) {
      return s.coordinate;
    });
  var a, o, u = t.map(function(s) {
    return s.coordinate === r && (a = !0), s.coordinate === n && (o = !0), s.coordinate;
  });
  return a || u.push(r), o || u.push(n), u;
}, Mt = function(t, r, n) {
  if (!t) return null;
  var i = t.scale, a = t.duplicateDomain, o = t.type, u = t.range, s = t.realScaleType === "scaleBand" ? i.bandwidth() / 2 : 2, c = (r || n) && o === "category" && i.bandwidth ? i.bandwidth() / s : 0;
  if (c = t.axisType === "angleAxis" && (u == null ? void 0 : u.length) >= 2 ? yt(u[0] - u[1]) * 2 * c : c, r && (t.ticks || t.niceTicks)) {
    var f = (t.ticks || t.niceTicks).map(function(l) {
      var d = a ? a.indexOf(l) : l;
      return {
        // If the scaleContent is not a number, the coordinate will be NaN.
        // That could be the case for example with a PointScale and a string as domain.
        coordinate: i(d) + c,
        value: l,
        offset: c
      };
    });
    return f.filter(function(l) {
      return !mi(l.coordinate);
    });
  }
  return t.isCategorical && t.categoricalDomain ? t.categoricalDomain.map(function(l, d) {
    return {
      coordinate: i(l) + c,
      value: l,
      index: d,
      offset: c
    };
  }) : i.ticks && !n ? i.ticks(t.tickCount).map(function(l) {
    return {
      coordinate: i(l) + c,
      value: l,
      offset: c
    };
  }) : i.domain().map(function(l, d) {
    return {
      coordinate: i(l) + c,
      value: a ? a[l] : l,
      index: d,
      offset: c
    };
  });
}, Kc = /* @__PURE__ */ new WeakMap(), Ni = function(t, r) {
  if (typeof r != "function")
    return t;
  Kc.has(t) || Kc.set(t, /* @__PURE__ */ new WeakMap());
  var n = Kc.get(t);
  if (n.has(r))
    return n.get(r);
  var i = function() {
    t.apply(void 0, arguments), r.apply(void 0, arguments);
  };
  return n.set(r, i), i;
}, zC = function(t, r, n) {
  var i = t.scale, a = t.type, o = t.layout, u = t.axisType;
  if (i === "auto")
    return o === "radial" && u === "radiusAxis" ? {
      scale: zn(),
      realScaleType: "band"
    } : o === "radial" && u === "angleAxis" ? {
      scale: la(),
      realScaleType: "linear"
    } : a === "category" && r && (r.indexOf("LineChart") >= 0 || r.indexOf("AreaChart") >= 0 || r.indexOf("ComposedChart") >= 0 && !n) ? {
      scale: In(),
      realScaleType: "point"
    } : a === "category" ? {
      scale: zn(),
      realScaleType: "band"
    } : {
      scale: la(),
      realScaleType: "linear"
    };
  if (yi(i)) {
    var s = "scale".concat(Va(i));
    return {
      scale: (Rm[s] || In)(),
      realScaleType: Rm[s] ? s : "point"
    };
  }
  return ue(i) ? {
    scale: i
  } : {
    scale: In(),
    realScaleType: "point"
  };
}, Xm = 1e-4, UC = function(t) {
  var r = t.domain();
  if (!(!r || r.length <= 2)) {
    var n = r.length, i = t.range(), a = Math.min(i[0], i[1]) - Xm, o = Math.max(i[0], i[1]) + Xm, u = t(r[0]), s = t(r[n - 1]);
    (u < a || u > o || s < a || s > o) && t.domain([r[0], r[n - 1]]);
  }
}, WC = function(t, r) {
  if (!t)
    return null;
  for (var n = 0, i = t.length; n < i; n++)
    if (t[n].item === r)
      return t[n].position;
  return null;
}, GC = function(t, r) {
  if (!r || r.length !== 2 || !K(r[0]) || !K(r[1]))
    return t;
  var n = Math.min(r[0], r[1]), i = Math.max(r[0], r[1]), a = [t[0], t[1]];
  return (!K(t[0]) || t[0] < n) && (a[0] = n), (!K(t[1]) || t[1] > i) && (a[1] = i), a[0] > i && (a[0] = i), a[1] < n && (a[1] = n), a;
}, HC = function(t) {
  var r = t.length;
  if (!(r <= 0))
    for (var n = 0, i = t[0].length; n < i; ++n)
      for (var a = 0, o = 0, u = 0; u < r; ++u) {
        var s = mi(t[u][n][1]) ? t[u][n][0] : t[u][n][1];
        s >= 0 ? (t[u][n][0] = a, t[u][n][1] = a + s, a = t[u][n][1]) : (t[u][n][0] = o, t[u][n][1] = o + s, o = t[u][n][1]);
      }
}, KC = function(t) {
  var r = t.length;
  if (!(r <= 0))
    for (var n = 0, i = t[0].length; n < i; ++n)
      for (var a = 0, o = 0; o < r; ++o) {
        var u = mi(t[o][n][1]) ? t[o][n][0] : t[o][n][1];
        u >= 0 ? (t[o][n][0] = a, t[o][n][1] = a + u, a = t[o][n][1]) : (t[o][n][0] = 0, t[o][n][1] = 0);
      }
}, VC = {
  sign: HC,
  // @ts-expect-error definitelytyped types are incorrect
  expand: q1,
  // @ts-expect-error definitelytyped types are incorrect
  none: Nr,
  // @ts-expect-error definitelytyped types are incorrect
  silhouette: L1,
  // @ts-expect-error definitelytyped types are incorrect
  wiggle: B1,
  positive: KC
}, YC = function(t, r, n) {
  var i = r.map(function(u) {
    return u.props.dataKey;
  }), a = VC[n], o = D1().keys(i).value(function(u, s) {
    return +tt(u, s, 0);
  }).order(Nl).offset(a);
  return o(t);
}, XC = function(t, r, n, i, a, o) {
  if (!t)
    return null;
  var u = o ? r.reverse() : r, s = {}, c = u.reduce(function(l, d) {
    var p, y = (p = d.type) !== null && p !== void 0 && p.defaultProps ? Te(Te({}, d.type.defaultProps), d.props) : d.props, v = y.stackId, h = y.hide;
    if (h)
      return l;
    var g = y[n], w = l[g] || {
      hasStack: !1,
      stackGroups: {}
    };
    if (ke(v)) {
      var b = w.stackGroups[v] || {
        numericAxisId: n,
        cateAxisId: i,
        items: []
      };
      b.items.push(d), w.hasStack = !0, w.stackGroups[v] = b;
    } else
      w.stackGroups[gi("_stackId_")] = {
        numericAxisId: n,
        cateAxisId: i,
        items: [d]
      };
    return Te(Te({}, l), {}, Rr({}, g, w));
  }, s), f = {};
  return Object.keys(c).reduce(function(l, d) {
    var p = c[d];
    if (p.hasStack) {
      var y = {};
      p.stackGroups = Object.keys(p.stackGroups).reduce(function(v, h) {
        var g = p.stackGroups[h];
        return Te(Te({}, v), {}, Rr({}, h, {
          numericAxisId: n,
          cateAxisId: i,
          items: g.items,
          stackedData: YC(t, g.items, a)
        }));
      }, y);
    }
    return Te(Te({}, l), {}, Rr({}, d, p));
  }, f);
}, ZC = function(t, r) {
  var n = r.realScaleType, i = r.type, a = r.tickCount, o = r.originalDomain, u = r.allowDecimals, s = n || r.scale;
  if (s !== "auto" && s !== "linear")
    return null;
  if (a && i === "number" && o && (o[0] === "auto" || o[1] === "auto")) {
    var c = t.domain();
    if (!c.length)
      return null;
    var f = sC(c, a, u);
    return t.domain([so(f), uo(f)]), {
      niceTicks: f
    };
  }
  if (a && i === "number") {
    var l = t.domain(), d = cC(l, a, u);
    return {
      niceTicks: d
    };
  }
  return null;
};
function Zm(e) {
  var t = e.axis, r = e.ticks, n = e.bandSize, i = e.entry, a = e.index, o = e.dataKey;
  if (t.type === "category") {
    if (!t.allowDuplicatedCategory && t.dataKey && !ce(i[t.dataKey])) {
      var u = Wi(r, "value", i[t.dataKey]);
      if (u)
        return u.coordinate + n / 2;
    }
    return r[a] ? r[a].coordinate + n / 2 : null;
  }
  var s = tt(i, ce(o) ? t.dataKey : o);
  return ce(s) ? null : t.scale(s);
}
var Jm = function(t) {
  var r = t.axis, n = t.ticks, i = t.offset, a = t.bandSize, o = t.entry, u = t.index;
  if (r.type === "category")
    return n[u] ? n[u].coordinate + i : null;
  var s = tt(o, r.dataKey, r.domain[u]);
  return ce(s) ? null : r.scale(s) - a / 2 + i;
}, JC = function(t) {
  var r = t.numericAxis, n = r.scale.domain();
  if (r.type === "number") {
    var i = Math.min(n[0], n[1]), a = Math.max(n[0], n[1]);
    return i <= 0 && a >= 0 ? 0 : a < 0 ? a : i;
  }
  return n[0];
}, QC = function(t, r) {
  var n, i = (n = t.type) !== null && n !== void 0 && n.defaultProps ? Te(Te({}, t.type.defaultProps), t.props) : t.props, a = i.stackId;
  if (ke(a)) {
    var o = r[a];
    if (o) {
      var u = o.items.indexOf(t);
      return u >= 0 ? o.stackedData[u] : null;
    }
  }
  return null;
}, eM = function(t) {
  return t.reduce(function(r, n) {
    return [so(n.concat([r[0]]).filter(K)), uo(n.concat([r[1]]).filter(K))];
  }, [1 / 0, -1 / 0]);
}, pw = function(t, r, n) {
  return Object.keys(t).reduce(function(i, a) {
    var o = t[a], u = o.stackedData, s = u.reduce(function(c, f) {
      var l = eM(f.slice(r, n + 1));
      return [Math.min(c[0], l[0]), Math.max(c[1], l[1])];
    }, [1 / 0, -1 / 0]);
    return [Math.min(s[0], i[0]), Math.max(s[1], i[1])];
  }, [1 / 0, -1 / 0]).map(function(i) {
    return i === 1 / 0 || i === -1 / 0 ? 0 : i;
  });
}, Qm = /^dataMin[\s]*-[\s]*([0-9]+([.]{1}[0-9]+){0,1})$/, eg = /^dataMax[\s]*\+[\s]*([0-9]+([.]{1}[0-9]+){0,1})$/, sf = function(t, r, n) {
  if (ue(t))
    return t(r, n);
  if (!Array.isArray(t))
    return r;
  var i = [];
  if (K(t[0]))
    i[0] = n ? t[0] : Math.min(t[0], r[0]);
  else if (Qm.test(t[0])) {
    var a = +Qm.exec(t[0])[1];
    i[0] = r[0] - a;
  } else ue(t[0]) ? i[0] = t[0](r[0]) : i[0] = r[0];
  if (K(t[1]))
    i[1] = n ? t[1] : Math.max(t[1], r[1]);
  else if (eg.test(t[1])) {
    var o = +eg.exec(t[1])[1];
    i[1] = r[1] + o;
  } else ue(t[1]) ? i[1] = t[1](r[1]) : i[1] = r[1];
  return i;
}, ma = function(t, r, n) {
  if (t && t.scale && t.scale.bandwidth) {
    var i = t.scale.bandwidth();
    if (!n || i > 0)
      return i;
  }
  if (t && r && r.length >= 2) {
    for (var a = md(r, function(l) {
      return l.coordinate;
    }), o = 1 / 0, u = 1, s = a.length; u < s; u++) {
      var c = a[u], f = a[u - 1];
      o = Math.min((c.coordinate || 0) - (f.coordinate || 0), o);
    }
    return o === 1 / 0 ? 0 : o;
  }
  return n ? void 0 : 0;
}, tg = function(t, r, n) {
  return !t || !t.length || co(t, at(n, "type.defaultProps.domain")) ? r : t;
}, hw = function(t, r) {
  var n = t.type.defaultProps ? Te(Te({}, t.type.defaultProps), t.props) : t.props, i = n.dataKey, a = n.name, o = n.unit, u = n.formatter, s = n.tooltipType, c = n.chartType, f = n.hide;
  return Te(Te({}, fe(t, !1)), {}, {
    dataKey: i,
    unit: o,
    formatter: u,
    name: a || i,
    color: Ud(t),
    value: tt(r, i),
    type: s,
    payload: r,
    chartType: c,
    hide: f
  });
};
function Zn(e) {
  "@babel/helpers - typeof";
  return Zn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Zn(e);
}
function rg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function ng(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? rg(Object(r), !0).forEach(function(n) {
      tM(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : rg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function tM(e, t, r) {
  return t = rM(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function rM(e) {
  var t = nM(e, "string");
  return Zn(t) == "symbol" ? t : t + "";
}
function nM(e, t) {
  if (Zn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Zn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var ga = Math.PI / 180, iM = function(t) {
  return t * 180 / Math.PI;
}, Fe = function(t, r, n, i) {
  return {
    x: t + Math.cos(-ga * i) * n,
    y: r + Math.sin(-ga * i) * n
  };
}, aM = function(t, r) {
  var n = t.x, i = t.y, a = r.x, o = r.y;
  return Math.sqrt(Math.pow(n - a, 2) + Math.pow(i - o, 2));
}, oM = function(t, r) {
  var n = t.x, i = t.y, a = r.cx, o = r.cy, u = aM({
    x: n,
    y: i
  }, {
    x: a,
    y: o
  });
  if (u <= 0)
    return {
      radius: u
    };
  var s = (n - a) / u, c = Math.acos(s);
  return i > o && (c = 2 * Math.PI - c), {
    radius: u,
    angle: iM(c),
    angleInRadian: c
  };
}, uM = function(t) {
  var r = t.startAngle, n = t.endAngle, i = Math.floor(r / 360), a = Math.floor(n / 360), o = Math.min(i, a);
  return {
    startAngle: r - o * 360,
    endAngle: n - o * 360
  };
}, sM = function(t, r) {
  var n = r.startAngle, i = r.endAngle, a = Math.floor(n / 360), o = Math.floor(i / 360), u = Math.min(a, o);
  return t + u * 360;
}, ig = function(t, r) {
  var n = t.x, i = t.y, a = oM({
    x: n,
    y: i
  }, r), o = a.radius, u = a.angle, s = r.innerRadius, c = r.outerRadius;
  if (o < s || o > c)
    return !1;
  if (o === 0)
    return !0;
  var f = uM(r), l = f.startAngle, d = f.endAngle, p = u, y;
  if (l <= d) {
    for (; p > d; )
      p -= 360;
    for (; p < l; )
      p += 360;
    y = p >= l && p <= d;
  } else {
    for (; p > l; )
      p -= 360;
    for (; p < d; )
      p += 360;
    y = p >= d && p <= l;
  }
  return y ? ng(ng({}, r), {}, {
    radius: o,
    angle: sM(p, r)
  }) : null;
};
function Jn(e) {
  "@babel/helpers - typeof";
  return Jn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Jn(e);
}
var cM = ["offset"];
function lM(e) {
  return hM(e) || pM(e) || dM(e) || fM();
}
function fM() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function dM(e, t) {
  if (e) {
    if (typeof e == "string") return cf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return cf(e, t);
  }
}
function pM(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function hM(e) {
  if (Array.isArray(e)) return cf(e);
}
function cf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function vM(e, t) {
  if (e == null) return {};
  var r = yM(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function yM(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function ag(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Re(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ag(Object(r), !0).forEach(function(n) {
      mM(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ag(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function mM(e, t, r) {
  return t = gM(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function gM(e) {
  var t = bM(e, "string");
  return Jn(t) == "symbol" ? t : t + "";
}
function bM(e, t) {
  if (Jn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Jn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function Qn() {
  return Qn = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Qn.apply(this, arguments);
}
var xM = function(t) {
  var r = t.value, n = t.formatter, i = ce(t.children) ? r : t.children;
  return ue(n) ? n(i) : i;
}, wM = function(t, r) {
  var n = yt(r - t), i = Math.min(Math.abs(r - t), 360);
  return n * i;
}, OM = function(t, r, n) {
  var i = t.position, a = t.viewBox, o = t.offset, u = t.className, s = a, c = s.cx, f = s.cy, l = s.innerRadius, d = s.outerRadius, p = s.startAngle, y = s.endAngle, v = s.clockWise, h = (l + d) / 2, g = wM(p, y), w = g >= 0 ? 1 : -1, b, O;
  i === "insideStart" ? (b = p + w * o, O = v) : i === "insideEnd" ? (b = y - w * o, O = !v) : i === "end" && (b = y + w * o, O = v), O = g <= 0 ? O : !O;
  var m = Fe(c, f, h, b), x = Fe(c, f, h, b + (O ? 1 : -1) * 359), _ = "M".concat(m.x, ",").concat(m.y, `
    A`).concat(h, ",").concat(h, ",0,1,").concat(O ? 0 : 1, `,
    `).concat(x.x, ",").concat(x.y), P = ce(t.id) ? gi("recharts-radial-line-") : t.id;
  return /* @__PURE__ */ T.createElement("text", Qn({}, n, {
    dominantBaseline: "central",
    className: pe("recharts-radial-bar-label", u)
  }), /* @__PURE__ */ T.createElement("defs", null, /* @__PURE__ */ T.createElement("path", {
    id: P,
    d: _
  })), /* @__PURE__ */ T.createElement("textPath", {
    xlinkHref: "#".concat(P)
  }, r));
}, _M = function(t) {
  var r = t.viewBox, n = t.offset, i = t.position, a = r, o = a.cx, u = a.cy, s = a.innerRadius, c = a.outerRadius, f = a.startAngle, l = a.endAngle, d = (f + l) / 2;
  if (i === "outside") {
    var p = Fe(o, u, c + n, d), y = p.x, v = p.y;
    return {
      x: y,
      y: v,
      textAnchor: y >= o ? "start" : "end",
      verticalAnchor: "middle"
    };
  }
  if (i === "center")
    return {
      x: o,
      y: u,
      textAnchor: "middle",
      verticalAnchor: "middle"
    };
  if (i === "centerTop")
    return {
      x: o,
      y: u,
      textAnchor: "middle",
      verticalAnchor: "start"
    };
  if (i === "centerBottom")
    return {
      x: o,
      y: u,
      textAnchor: "middle",
      verticalAnchor: "end"
    };
  var h = (s + c) / 2, g = Fe(o, u, h, d), w = g.x, b = g.y;
  return {
    x: w,
    y: b,
    textAnchor: "middle",
    verticalAnchor: "middle"
  };
}, SM = function(t) {
  var r = t.viewBox, n = t.parentViewBox, i = t.offset, a = t.position, o = r, u = o.x, s = o.y, c = o.width, f = o.height, l = f >= 0 ? 1 : -1, d = l * i, p = l > 0 ? "end" : "start", y = l > 0 ? "start" : "end", v = c >= 0 ? 1 : -1, h = v * i, g = v > 0 ? "end" : "start", w = v > 0 ? "start" : "end";
  if (a === "top") {
    var b = {
      x: u + c / 2,
      y: s - l * i,
      textAnchor: "middle",
      verticalAnchor: p
    };
    return Re(Re({}, b), n ? {
      height: Math.max(s - n.y, 0),
      width: c
    } : {});
  }
  if (a === "bottom") {
    var O = {
      x: u + c / 2,
      y: s + f + d,
      textAnchor: "middle",
      verticalAnchor: y
    };
    return Re(Re({}, O), n ? {
      height: Math.max(n.y + n.height - (s + f), 0),
      width: c
    } : {});
  }
  if (a === "left") {
    var m = {
      x: u - h,
      y: s + f / 2,
      textAnchor: g,
      verticalAnchor: "middle"
    };
    return Re(Re({}, m), n ? {
      width: Math.max(m.x - n.x, 0),
      height: f
    } : {});
  }
  if (a === "right") {
    var x = {
      x: u + c + h,
      y: s + f / 2,
      textAnchor: w,
      verticalAnchor: "middle"
    };
    return Re(Re({}, x), n ? {
      width: Math.max(n.x + n.width - x.x, 0),
      height: f
    } : {});
  }
  var _ = n ? {
    width: c,
    height: f
  } : {};
  return a === "insideLeft" ? Re({
    x: u + h,
    y: s + f / 2,
    textAnchor: w,
    verticalAnchor: "middle"
  }, _) : a === "insideRight" ? Re({
    x: u + c - h,
    y: s + f / 2,
    textAnchor: g,
    verticalAnchor: "middle"
  }, _) : a === "insideTop" ? Re({
    x: u + c / 2,
    y: s + d,
    textAnchor: "middle",
    verticalAnchor: y
  }, _) : a === "insideBottom" ? Re({
    x: u + c / 2,
    y: s + f - d,
    textAnchor: "middle",
    verticalAnchor: p
  }, _) : a === "insideTopLeft" ? Re({
    x: u + h,
    y: s + d,
    textAnchor: w,
    verticalAnchor: y
  }, _) : a === "insideTopRight" ? Re({
    x: u + c - h,
    y: s + d,
    textAnchor: g,
    verticalAnchor: y
  }, _) : a === "insideBottomLeft" ? Re({
    x: u + h,
    y: s + f - d,
    textAnchor: w,
    verticalAnchor: p
  }, _) : a === "insideBottomRight" ? Re({
    x: u + c - h,
    y: s + f - d,
    textAnchor: g,
    verticalAnchor: p
  }, _) : sn(a) && (K(a.x) || Al(a.x)) && (K(a.y) || Al(a.y)) ? Re({
    x: u + hr(a.x, c),
    y: s + hr(a.y, f),
    textAnchor: "end",
    verticalAnchor: "end"
  }, _) : Re({
    x: u + c / 2,
    y: s + f / 2,
    textAnchor: "middle",
    verticalAnchor: "middle"
  }, _);
}, PM = function(t) {
  return "cx" in t && K(t.cx);
};
function Ue(e) {
  var t = e.offset, r = t === void 0 ? 5 : t, n = vM(e, cM), i = Re({
    offset: r
  }, n), a = i.viewBox, o = i.position, u = i.value, s = i.children, c = i.content, f = i.className, l = f === void 0 ? "" : f, d = i.textBreakAll;
  if (!a || ce(u) && ce(s) && !/* @__PURE__ */ xt(c) && !ue(c))
    return null;
  if (/* @__PURE__ */ xt(c))
    return /* @__PURE__ */ De(c, i);
  var p;
  if (ue(c)) {
    if (p = /* @__PURE__ */ a0(c, i), /* @__PURE__ */ xt(p))
      return p;
  } else
    p = xM(i);
  var y = PM(a), v = fe(i, !0);
  if (y && (o === "insideStart" || o === "insideEnd" || o === "end"))
    return OM(i, p, v);
  var h = y ? _M(i) : SM(i);
  return /* @__PURE__ */ T.createElement(na, Qn({
    className: pe("recharts-label", l)
  }, v, h, {
    breakAll: d
  }), p);
}
Ue.displayName = "Label";
var vw = function(t) {
  var r = t.cx, n = t.cy, i = t.angle, a = t.startAngle, o = t.endAngle, u = t.r, s = t.radius, c = t.innerRadius, f = t.outerRadius, l = t.x, d = t.y, p = t.top, y = t.left, v = t.width, h = t.height, g = t.clockWise, w = t.labelViewBox;
  if (w)
    return w;
  if (K(v) && K(h)) {
    if (K(l) && K(d))
      return {
        x: l,
        y: d,
        width: v,
        height: h
      };
    if (K(p) && K(y))
      return {
        x: p,
        y,
        width: v,
        height: h
      };
  }
  return K(l) && K(d) ? {
    x: l,
    y: d,
    width: 0,
    height: 0
  } : K(r) && K(n) ? {
    cx: r,
    cy: n,
    startAngle: a || i || 0,
    endAngle: o || i || 0,
    innerRadius: c || 0,
    outerRadius: f || s || u || 0,
    clockWise: g
  } : t.viewBox ? t.viewBox : {};
}, AM = function(t, r) {
  return t ? t === !0 ? /* @__PURE__ */ T.createElement(Ue, {
    key: "label-implicit",
    viewBox: r
  }) : ke(t) ? /* @__PURE__ */ T.createElement(Ue, {
    key: "label-implicit",
    viewBox: r,
    value: t
  }) : /* @__PURE__ */ xt(t) ? t.type === Ue ? /* @__PURE__ */ De(t, {
    key: "label-implicit",
    viewBox: r
  }) : /* @__PURE__ */ T.createElement(Ue, {
    key: "label-implicit",
    content: t,
    viewBox: r
  }) : ue(t) ? /* @__PURE__ */ T.createElement(Ue, {
    key: "label-implicit",
    content: t,
    viewBox: r
  }) : sn(t) ? /* @__PURE__ */ T.createElement(Ue, Qn({
    viewBox: r
  }, t, {
    key: "label-implicit"
  })) : null : null;
}, EM = function(t, r) {
  var n = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : !0;
  if (!t || !t.children && n && !t.label)
    return null;
  var i = t.children, a = vw(t), o = ot(i, Ue).map(function(s, c) {
    return /* @__PURE__ */ De(s, {
      viewBox: r || a,
      // eslint-disable-next-line react/no-array-index-key
      key: "label-".concat(c)
    });
  });
  if (!n)
    return o;
  var u = AM(t.label, r || a);
  return [u].concat(lM(o));
};
Ue.parseViewBox = vw;
Ue.renderCallByParent = EM;
var Vc, og;
function TM() {
  if (og) return Vc;
  og = 1;
  function e(t) {
    var r = t == null ? 0 : t.length;
    return r ? t[r - 1] : void 0;
  }
  return Vc = e, Vc;
}
var jM = TM();
const CM = /* @__PURE__ */ Pe(jM);
function ei(e) {
  "@babel/helpers - typeof";
  return ei = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ei(e);
}
var MM = ["valueAccessor"], IM = ["data", "dataKey", "clockWise", "id", "textBreakAll"];
function $M(e) {
  return DM(e) || NM(e) || kM(e) || RM();
}
function RM() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function kM(e, t) {
  if (e) {
    if (typeof e == "string") return lf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return lf(e, t);
  }
}
function NM(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function DM(e) {
  if (Array.isArray(e)) return lf(e);
}
function lf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function ba() {
  return ba = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, ba.apply(this, arguments);
}
function ug(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function sg(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ug(Object(r), !0).forEach(function(n) {
      qM(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ug(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function qM(e, t, r) {
  return t = LM(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function LM(e) {
  var t = BM(e, "string");
  return ei(t) == "symbol" ? t : t + "";
}
function BM(e, t) {
  if (ei(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ei(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function cg(e, t) {
  if (e == null) return {};
  var r = FM(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function FM(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
var zM = function(t) {
  return Array.isArray(t.value) ? CM(t.value) : t.value;
};
function Vt(e) {
  var t = e.valueAccessor, r = t === void 0 ? zM : t, n = cg(e, MM), i = n.data, a = n.dataKey, o = n.clockWise, u = n.id, s = n.textBreakAll, c = cg(n, IM);
  return !i || !i.length ? null : /* @__PURE__ */ T.createElement(je, {
    className: "recharts-label-list"
  }, i.map(function(f, l) {
    var d = ce(a) ? r(f, l) : tt(f && f.payload, a), p = ce(u) ? {} : {
      id: "".concat(u, "-").concat(l)
    };
    return /* @__PURE__ */ T.createElement(Ue, ba({}, fe(f, !0), c, p, {
      parentViewBox: f.parentViewBox,
      value: d,
      textBreakAll: s,
      viewBox: Ue.parseViewBox(ce(o) ? f : sg(sg({}, f), {}, {
        clockWise: o
      })),
      key: "label-".concat(l),
      index: l
    }));
  }));
}
Vt.displayName = "LabelList";
function UM(e, t) {
  return e ? e === !0 ? /* @__PURE__ */ T.createElement(Vt, {
    key: "labelList-implicit",
    data: t
  }) : /* @__PURE__ */ T.isValidElement(e) || ue(e) ? /* @__PURE__ */ T.createElement(Vt, {
    key: "labelList-implicit",
    data: t,
    content: e
  }) : sn(e) ? /* @__PURE__ */ T.createElement(Vt, ba({
    data: t
  }, e, {
    key: "labelList-implicit"
  })) : null : null;
}
function WM(e, t) {
  var r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : !0;
  if (!e || !e.children && r && !e.label)
    return null;
  var n = e.children, i = ot(n, Vt).map(function(o, u) {
    return /* @__PURE__ */ De(o, {
      data: t,
      // eslint-disable-next-line react/no-array-index-key
      key: "labelList-".concat(u)
    });
  });
  if (!r)
    return i;
  var a = UM(e.label, t);
  return [a].concat($M(i));
}
Vt.renderCallByParent = WM;
function ti(e) {
  "@babel/helpers - typeof";
  return ti = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ti(e);
}
function ff() {
  return ff = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, ff.apply(this, arguments);
}
function lg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function fg(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? lg(Object(r), !0).forEach(function(n) {
      GM(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : lg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function GM(e, t, r) {
  return t = HM(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function HM(e) {
  var t = KM(e, "string");
  return ti(t) == "symbol" ? t : t + "";
}
function KM(e, t) {
  if (ti(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ti(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var VM = function(t, r) {
  var n = yt(r - t), i = Math.min(Math.abs(r - t), 359.999);
  return n * i;
}, Di = function(t) {
  var r = t.cx, n = t.cy, i = t.radius, a = t.angle, o = t.sign, u = t.isExternal, s = t.cornerRadius, c = t.cornerIsExternal, f = s * (u ? 1 : -1) + i, l = Math.asin(s / f) / ga, d = c ? a : a + o * l, p = Fe(r, n, f, d), y = Fe(r, n, i, d), v = c ? a - o * l : a, h = Fe(r, n, f * Math.cos(l * ga), v);
  return {
    center: p,
    circleTangency: y,
    lineTangency: h,
    theta: l
  };
}, yw = function(t) {
  var r = t.cx, n = t.cy, i = t.innerRadius, a = t.outerRadius, o = t.startAngle, u = t.endAngle, s = VM(o, u), c = o + s, f = Fe(r, n, a, o), l = Fe(r, n, a, c), d = "M ".concat(f.x, ",").concat(f.y, `
    A `).concat(a, ",").concat(a, `,0,
    `).concat(+(Math.abs(s) > 180), ",").concat(+(o > c), `,
    `).concat(l.x, ",").concat(l.y, `
  `);
  if (i > 0) {
    var p = Fe(r, n, i, o), y = Fe(r, n, i, c);
    d += "L ".concat(y.x, ",").concat(y.y, `
            A `).concat(i, ",").concat(i, `,0,
            `).concat(+(Math.abs(s) > 180), ",").concat(+(o <= c), `,
            `).concat(p.x, ",").concat(p.y, " Z");
  } else
    d += "L ".concat(r, ",").concat(n, " Z");
  return d;
}, YM = function(t) {
  var r = t.cx, n = t.cy, i = t.innerRadius, a = t.outerRadius, o = t.cornerRadius, u = t.forceCornerRadius, s = t.cornerIsExternal, c = t.startAngle, f = t.endAngle, l = yt(f - c), d = Di({
    cx: r,
    cy: n,
    radius: a,
    angle: c,
    sign: l,
    cornerRadius: o,
    cornerIsExternal: s
  }), p = d.circleTangency, y = d.lineTangency, v = d.theta, h = Di({
    cx: r,
    cy: n,
    radius: a,
    angle: f,
    sign: -l,
    cornerRadius: o,
    cornerIsExternal: s
  }), g = h.circleTangency, w = h.lineTangency, b = h.theta, O = s ? Math.abs(c - f) : Math.abs(c - f) - v - b;
  if (O < 0)
    return u ? "M ".concat(y.x, ",").concat(y.y, `
        a`).concat(o, ",").concat(o, ",0,0,1,").concat(o * 2, `,0
        a`).concat(o, ",").concat(o, ",0,0,1,").concat(-o * 2, `,0
      `) : yw({
      cx: r,
      cy: n,
      innerRadius: i,
      outerRadius: a,
      startAngle: c,
      endAngle: f
    });
  var m = "M ".concat(y.x, ",").concat(y.y, `
    A`).concat(o, ",").concat(o, ",0,0,").concat(+(l < 0), ",").concat(p.x, ",").concat(p.y, `
    A`).concat(a, ",").concat(a, ",0,").concat(+(O > 180), ",").concat(+(l < 0), ",").concat(g.x, ",").concat(g.y, `
    A`).concat(o, ",").concat(o, ",0,0,").concat(+(l < 0), ",").concat(w.x, ",").concat(w.y, `
  `);
  if (i > 0) {
    var x = Di({
      cx: r,
      cy: n,
      radius: i,
      angle: c,
      sign: l,
      isExternal: !0,
      cornerRadius: o,
      cornerIsExternal: s
    }), _ = x.circleTangency, P = x.lineTangency, E = x.theta, I = Di({
      cx: r,
      cy: n,
      radius: i,
      angle: f,
      sign: -l,
      isExternal: !0,
      cornerRadius: o,
      cornerIsExternal: s
    }), S = I.circleTangency, j = I.lineTangency, M = I.theta, R = s ? Math.abs(c - f) : Math.abs(c - f) - E - M;
    if (R < 0 && o === 0)
      return "".concat(m, "L").concat(r, ",").concat(n, "Z");
    m += "L".concat(j.x, ",").concat(j.y, `
      A`).concat(o, ",").concat(o, ",0,0,").concat(+(l < 0), ",").concat(S.x, ",").concat(S.y, `
      A`).concat(i, ",").concat(i, ",0,").concat(+(R > 180), ",").concat(+(l > 0), ",").concat(_.x, ",").concat(_.y, `
      A`).concat(o, ",").concat(o, ",0,0,").concat(+(l < 0), ",").concat(P.x, ",").concat(P.y, "Z");
  } else
    m += "L".concat(r, ",").concat(n, "Z");
  return m;
}, XM = {
  cx: 0,
  cy: 0,
  innerRadius: 0,
  outerRadius: 0,
  startAngle: 0,
  endAngle: 0,
  cornerRadius: 0,
  forceCornerRadius: !1,
  cornerIsExternal: !1
}, mw = function(t) {
  var r = fg(fg({}, XM), t), n = r.cx, i = r.cy, a = r.innerRadius, o = r.outerRadius, u = r.cornerRadius, s = r.forceCornerRadius, c = r.cornerIsExternal, f = r.startAngle, l = r.endAngle, d = r.className;
  if (o < a || f === l)
    return null;
  var p = pe("recharts-sector", d), y = o - a, v = hr(u, y, 0, !0), h;
  return v > 0 && Math.abs(f - l) < 360 ? h = YM({
    cx: n,
    cy: i,
    innerRadius: a,
    outerRadius: o,
    cornerRadius: Math.min(v, y / 2),
    forceCornerRadius: s,
    cornerIsExternal: c,
    startAngle: f,
    endAngle: l
  }) : h = yw({
    cx: n,
    cy: i,
    innerRadius: a,
    outerRadius: o,
    startAngle: f,
    endAngle: l
  }), /* @__PURE__ */ T.createElement("path", ff({}, fe(r, !0), {
    className: p,
    d: h,
    role: "img"
  }));
};
function ri(e) {
  "@babel/helpers - typeof";
  return ri = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ri(e);
}
function df() {
  return df = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, df.apply(this, arguments);
}
function dg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function pg(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? dg(Object(r), !0).forEach(function(n) {
      ZM(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : dg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function ZM(e, t, r) {
  return t = JM(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function JM(e) {
  var t = QM(e, "string");
  return ri(t) == "symbol" ? t : t + "";
}
function QM(e, t) {
  if (ri(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ri(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var hg = {
  curveBasisClosed: A1,
  curveBasisOpen: E1,
  curveBasis: P1,
  curveBumpX: f1,
  curveBumpY: d1,
  curveLinearClosed: T1,
  curveLinear: Xa,
  curveMonotoneX: j1,
  curveMonotoneY: C1,
  curveNatural: M1,
  curveStep: I1,
  curveStepAfter: R1,
  curveStepBefore: $1
}, qi = function(t) {
  return t.x === +t.x && t.y === +t.y;
}, Sn = function(t) {
  return t.x;
}, Pn = function(t) {
  return t.y;
}, eI = function(t, r) {
  if (ue(t))
    return t;
  var n = "curve".concat(Va(t));
  return (n === "curveMonotone" || n === "curveBump") && r ? hg["".concat(n).concat(r === "vertical" ? "Y" : "X")] : hg[n] || Xa;
}, tI = function(t) {
  var r = t.type, n = r === void 0 ? "linear" : r, i = t.points, a = i === void 0 ? [] : i, o = t.baseLine, u = t.layout, s = t.connectNulls, c = s === void 0 ? !1 : s, f = eI(n, u), l = c ? a.filter(function(v) {
    return qi(v);
  }) : a, d;
  if (Array.isArray(o)) {
    var p = c ? o.filter(function(v) {
      return qi(v);
    }) : o, y = l.map(function(v, h) {
      return pg(pg({}, v), {}, {
        base: p[h]
      });
    });
    return u === "vertical" ? d = Ci().y(Pn).x1(Sn).x0(function(v) {
      return v.base.x;
    }) : d = Ci().x(Sn).y1(Pn).y0(function(v) {
      return v.base.y;
    }), d.defined(qi).curve(f), d(y);
  }
  return u === "vertical" && K(o) ? d = Ci().y(Pn).x1(Sn).x0(o) : K(o) ? d = Ci().x(Sn).y1(Pn).y0(o) : d = O0().x(Sn).y(Pn), d.defined(qi).curve(f), d(l);
}, pf = function(t) {
  var r = t.className, n = t.points, i = t.path, a = t.pathRef;
  if ((!n || !n.length) && !i)
    return null;
  var o = n && n.length ? tI(t) : i;
  return /* @__PURE__ */ T.createElement("path", df({}, fe(t, !1), Gi(t), {
    className: pe("recharts-curve", r),
    d: o,
    ref: a
  }));
}, Li = { exports: {} }, Bi = { exports: {} }, me = {};
/** @license React v16.13.1
 * react-is.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var vg;
function rI() {
  if (vg) return me;
  vg = 1;
  var e = typeof Symbol == "function" && Symbol.for, t = e ? Symbol.for("react.element") : 60103, r = e ? Symbol.for("react.portal") : 60106, n = e ? Symbol.for("react.fragment") : 60107, i = e ? Symbol.for("react.strict_mode") : 60108, a = e ? Symbol.for("react.profiler") : 60114, o = e ? Symbol.for("react.provider") : 60109, u = e ? Symbol.for("react.context") : 60110, s = e ? Symbol.for("react.async_mode") : 60111, c = e ? Symbol.for("react.concurrent_mode") : 60111, f = e ? Symbol.for("react.forward_ref") : 60112, l = e ? Symbol.for("react.suspense") : 60113, d = e ? Symbol.for("react.suspense_list") : 60120, p = e ? Symbol.for("react.memo") : 60115, y = e ? Symbol.for("react.lazy") : 60116, v = e ? Symbol.for("react.block") : 60121, h = e ? Symbol.for("react.fundamental") : 60117, g = e ? Symbol.for("react.responder") : 60118, w = e ? Symbol.for("react.scope") : 60119;
  function b(m) {
    if (typeof m == "object" && m !== null) {
      var x = m.$$typeof;
      switch (x) {
        case t:
          switch (m = m.type, m) {
            case s:
            case c:
            case n:
            case a:
            case i:
            case l:
              return m;
            default:
              switch (m = m && m.$$typeof, m) {
                case u:
                case f:
                case y:
                case p:
                case o:
                  return m;
                default:
                  return x;
              }
          }
        case r:
          return x;
      }
    }
  }
  function O(m) {
    return b(m) === c;
  }
  return me.AsyncMode = s, me.ConcurrentMode = c, me.ContextConsumer = u, me.ContextProvider = o, me.Element = t, me.ForwardRef = f, me.Fragment = n, me.Lazy = y, me.Memo = p, me.Portal = r, me.Profiler = a, me.StrictMode = i, me.Suspense = l, me.isAsyncMode = function(m) {
    return O(m) || b(m) === s;
  }, me.isConcurrentMode = O, me.isContextConsumer = function(m) {
    return b(m) === u;
  }, me.isContextProvider = function(m) {
    return b(m) === o;
  }, me.isElement = function(m) {
    return typeof m == "object" && m !== null && m.$$typeof === t;
  }, me.isForwardRef = function(m) {
    return b(m) === f;
  }, me.isFragment = function(m) {
    return b(m) === n;
  }, me.isLazy = function(m) {
    return b(m) === y;
  }, me.isMemo = function(m) {
    return b(m) === p;
  }, me.isPortal = function(m) {
    return b(m) === r;
  }, me.isProfiler = function(m) {
    return b(m) === a;
  }, me.isStrictMode = function(m) {
    return b(m) === i;
  }, me.isSuspense = function(m) {
    return b(m) === l;
  }, me.isValidElementType = function(m) {
    return typeof m == "string" || typeof m == "function" || m === n || m === c || m === a || m === i || m === l || m === d || typeof m == "object" && m !== null && (m.$$typeof === y || m.$$typeof === p || m.$$typeof === o || m.$$typeof === u || m.$$typeof === f || m.$$typeof === h || m.$$typeof === g || m.$$typeof === w || m.$$typeof === v);
  }, me.typeOf = b, me;
}
var ge = {};
/** @license React v16.13.1
 * react-is.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var yg;
function nI() {
  return yg || (yg = 1, process.env.NODE_ENV !== "production" && function() {
    var e = typeof Symbol == "function" && Symbol.for, t = e ? Symbol.for("react.element") : 60103, r = e ? Symbol.for("react.portal") : 60106, n = e ? Symbol.for("react.fragment") : 60107, i = e ? Symbol.for("react.strict_mode") : 60108, a = e ? Symbol.for("react.profiler") : 60114, o = e ? Symbol.for("react.provider") : 60109, u = e ? Symbol.for("react.context") : 60110, s = e ? Symbol.for("react.async_mode") : 60111, c = e ? Symbol.for("react.concurrent_mode") : 60111, f = e ? Symbol.for("react.forward_ref") : 60112, l = e ? Symbol.for("react.suspense") : 60113, d = e ? Symbol.for("react.suspense_list") : 60120, p = e ? Symbol.for("react.memo") : 60115, y = e ? Symbol.for("react.lazy") : 60116, v = e ? Symbol.for("react.block") : 60121, h = e ? Symbol.for("react.fundamental") : 60117, g = e ? Symbol.for("react.responder") : 60118, w = e ? Symbol.for("react.scope") : 60119;
    function b(C) {
      return typeof C == "string" || typeof C == "function" || // Note: its typeof might be other than 'symbol' or 'number' if it's a polyfill.
      C === n || C === c || C === a || C === i || C === l || C === d || typeof C == "object" && C !== null && (C.$$typeof === y || C.$$typeof === p || C.$$typeof === o || C.$$typeof === u || C.$$typeof === f || C.$$typeof === h || C.$$typeof === g || C.$$typeof === w || C.$$typeof === v);
    }
    function O(C) {
      if (typeof C == "object" && C !== null) {
        var se = C.$$typeof;
        switch (se) {
          case t:
            var W = C.type;
            switch (W) {
              case s:
              case c:
              case n:
              case a:
              case i:
              case l:
                return W;
              default:
                var he = W && W.$$typeof;
                switch (he) {
                  case u:
                  case f:
                  case y:
                  case p:
                  case o:
                    return he;
                  default:
                    return se;
                }
            }
          case r:
            return se;
        }
      }
    }
    var m = s, x = c, _ = u, P = o, E = t, I = f, S = n, j = y, M = p, R = r, k = a, q = i, L = l, U = !1;
    function z(C) {
      return U || (U = !0, console.warn("The ReactIs.isAsyncMode() alias has been deprecated, and will be removed in React 17+. Update your code to use ReactIs.isConcurrentMode() instead. It has the exact same API.")), $(C) || O(C) === s;
    }
    function $(C) {
      return O(C) === c;
    }
    function D(C) {
      return O(C) === u;
    }
    function B(C) {
      return O(C) === o;
    }
    function G(C) {
      return typeof C == "object" && C !== null && C.$$typeof === t;
    }
    function V(C) {
      return O(C) === f;
    }
    function te(C) {
      return O(C) === n;
    }
    function re(C) {
      return O(C) === y;
    }
    function ae(C) {
      return O(C) === p;
    }
    function ne(C) {
      return O(C) === r;
    }
    function F(C) {
      return O(C) === a;
    }
    function H(C) {
      return O(C) === i;
    }
    function ee(C) {
      return O(C) === l;
    }
    ge.AsyncMode = m, ge.ConcurrentMode = x, ge.ContextConsumer = _, ge.ContextProvider = P, ge.Element = E, ge.ForwardRef = I, ge.Fragment = S, ge.Lazy = j, ge.Memo = M, ge.Portal = R, ge.Profiler = k, ge.StrictMode = q, ge.Suspense = L, ge.isAsyncMode = z, ge.isConcurrentMode = $, ge.isContextConsumer = D, ge.isContextProvider = B, ge.isElement = G, ge.isForwardRef = V, ge.isFragment = te, ge.isLazy = re, ge.isMemo = ae, ge.isPortal = ne, ge.isProfiler = F, ge.isStrictMode = H, ge.isSuspense = ee, ge.isValidElementType = b, ge.typeOf = O;
  }()), ge;
}
var mg;
function gw() {
  return mg || (mg = 1, process.env.NODE_ENV === "production" ? Bi.exports = rI() : Bi.exports = nI()), Bi.exports;
}
/*
object-assign
(c) Sindre Sorhus
@license MIT
*/
var Yc, gg;
function iI() {
  if (gg) return Yc;
  gg = 1;
  var e = Object.getOwnPropertySymbols, t = Object.prototype.hasOwnProperty, r = Object.prototype.propertyIsEnumerable;
  function n(a) {
    if (a == null)
      throw new TypeError("Object.assign cannot be called with null or undefined");
    return Object(a);
  }
  function i() {
    try {
      if (!Object.assign)
        return !1;
      var a = new String("abc");
      if (a[5] = "de", Object.getOwnPropertyNames(a)[0] === "5")
        return !1;
      for (var o = {}, u = 0; u < 10; u++)
        o["_" + String.fromCharCode(u)] = u;
      var s = Object.getOwnPropertyNames(o).map(function(f) {
        return o[f];
      });
      if (s.join("") !== "0123456789")
        return !1;
      var c = {};
      return "abcdefghijklmnopqrst".split("").forEach(function(f) {
        c[f] = f;
      }), Object.keys(Object.assign({}, c)).join("") === "abcdefghijklmnopqrst";
    } catch {
      return !1;
    }
  }
  return Yc = i() ? Object.assign : function(a, o) {
    for (var u, s = n(a), c, f = 1; f < arguments.length; f++) {
      u = Object(arguments[f]);
      for (var l in u)
        t.call(u, l) && (s[l] = u[l]);
      if (e) {
        c = e(u);
        for (var d = 0; d < c.length; d++)
          r.call(u, c[d]) && (s[c[d]] = u[c[d]]);
      }
    }
    return s;
  }, Yc;
}
var Xc, bg;
function Wd() {
  if (bg) return Xc;
  bg = 1;
  var e = "SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED";
  return Xc = e, Xc;
}
var Zc, xg;
function bw() {
  return xg || (xg = 1, Zc = Function.call.bind(Object.prototype.hasOwnProperty)), Zc;
}
var Jc, wg;
function aI() {
  if (wg) return Jc;
  wg = 1;
  var e = function() {
  };
  if (process.env.NODE_ENV !== "production") {
    var t = /* @__PURE__ */ Wd(), r = {}, n = /* @__PURE__ */ bw();
    e = function(a) {
      var o = "Warning: " + a;
      typeof console < "u" && console.error(o);
      try {
        throw new Error(o);
      } catch {
      }
    };
  }
  function i(a, o, u, s, c) {
    if (process.env.NODE_ENV !== "production") {
      for (var f in a)
        if (n(a, f)) {
          var l;
          try {
            if (typeof a[f] != "function") {
              var d = Error(
                (s || "React class") + ": " + u + " type `" + f + "` is invalid; it must be a function, usually from the `prop-types` package, but received `" + typeof a[f] + "`.This often happens because of typos such as `PropTypes.function` instead of `PropTypes.func`."
              );
              throw d.name = "Invariant Violation", d;
            }
            l = a[f](o, f, s, u, null, t);
          } catch (y) {
            l = y;
          }
          if (l && !(l instanceof Error) && e(
            (s || "React class") + ": type specification of " + u + " `" + f + "` is invalid; the type checker function must return `null` or an `Error` but returned a " + typeof l + ". You may have forgotten to pass an argument to the type checker creator (arrayOf, instanceOf, objectOf, oneOf, oneOfType, and shape all require an argument)."
          ), l instanceof Error && !(l.message in r)) {
            r[l.message] = !0;
            var p = c ? c() : "";
            e(
              "Failed " + u + " type: " + l.message + (p ?? "")
            );
          }
        }
    }
  }
  return i.resetWarningCache = function() {
    process.env.NODE_ENV !== "production" && (r = {});
  }, Jc = i, Jc;
}
var Qc, Og;
function oI() {
  if (Og) return Qc;
  Og = 1;
  var e = gw(), t = iI(), r = /* @__PURE__ */ Wd(), n = /* @__PURE__ */ bw(), i = /* @__PURE__ */ aI(), a = function() {
  };
  process.env.NODE_ENV !== "production" && (a = function(u) {
    var s = "Warning: " + u;
    typeof console < "u" && console.error(s);
    try {
      throw new Error(s);
    } catch {
    }
  });
  function o() {
    return null;
  }
  return Qc = function(u, s) {
    var c = typeof Symbol == "function" && Symbol.iterator, f = "@@iterator";
    function l($) {
      var D = $ && (c && $[c] || $[f]);
      if (typeof D == "function")
        return D;
    }
    var d = "<<anonymous>>", p = {
      array: g("array"),
      bigint: g("bigint"),
      bool: g("boolean"),
      func: g("function"),
      number: g("number"),
      object: g("object"),
      string: g("string"),
      symbol: g("symbol"),
      any: w(),
      arrayOf: b,
      element: O(),
      elementType: m(),
      instanceOf: x,
      node: I(),
      objectOf: P,
      oneOf: _,
      oneOfType: E,
      shape: j,
      exact: M
    };
    function y($, D) {
      return $ === D ? $ !== 0 || 1 / $ === 1 / D : $ !== $ && D !== D;
    }
    function v($, D) {
      this.message = $, this.data = D && typeof D == "object" ? D : {}, this.stack = "";
    }
    v.prototype = Error.prototype;
    function h($) {
      if (process.env.NODE_ENV !== "production")
        var D = {}, B = 0;
      function G(te, re, ae, ne, F, H, ee) {
        if (ne = ne || d, H = H || ae, ee !== r) {
          if (s) {
            var C = new Error(
              "Calling PropTypes validators directly is not supported by the `prop-types` package. Use `PropTypes.checkPropTypes()` to call them. Read more at http://fb.me/use-check-prop-types"
            );
            throw C.name = "Invariant Violation", C;
          } else if (process.env.NODE_ENV !== "production" && typeof console < "u") {
            var se = ne + ":" + ae;
            !D[se] && // Avoid spamming the console because they are often not actionable except for lib authors
            B < 3 && (a(
              "You are manually calling a React.PropTypes validation function for the `" + H + "` prop on `" + ne + "`. This is deprecated and will throw in the standalone `prop-types` package. You may be seeing this warning due to a third-party PropTypes library. See https://fb.me/react-warning-dont-call-proptypes for details."
            ), D[se] = !0, B++);
          }
        }
        return re[ae] == null ? te ? re[ae] === null ? new v("The " + F + " `" + H + "` is marked as required " + ("in `" + ne + "`, but its value is `null`.")) : new v("The " + F + " `" + H + "` is marked as required in " + ("`" + ne + "`, but its value is `undefined`.")) : null : $(re, ae, ne, F, H);
      }
      var V = G.bind(null, !1);
      return V.isRequired = G.bind(null, !0), V;
    }
    function g($) {
      function D(B, G, V, te, re, ae) {
        var ne = B[G], F = q(ne);
        if (F !== $) {
          var H = L(ne);
          return new v(
            "Invalid " + te + " `" + re + "` of type " + ("`" + H + "` supplied to `" + V + "`, expected ") + ("`" + $ + "`."),
            { expectedType: $ }
          );
        }
        return null;
      }
      return h(D);
    }
    function w() {
      return h(o);
    }
    function b($) {
      function D(B, G, V, te, re) {
        if (typeof $ != "function")
          return new v("Property `" + re + "` of component `" + V + "` has invalid PropType notation inside arrayOf.");
        var ae = B[G];
        if (!Array.isArray(ae)) {
          var ne = q(ae);
          return new v("Invalid " + te + " `" + re + "` of type " + ("`" + ne + "` supplied to `" + V + "`, expected an array."));
        }
        for (var F = 0; F < ae.length; F++) {
          var H = $(ae, F, V, te, re + "[" + F + "]", r);
          if (H instanceof Error)
            return H;
        }
        return null;
      }
      return h(D);
    }
    function O() {
      function $(D, B, G, V, te) {
        var re = D[B];
        if (!u(re)) {
          var ae = q(re);
          return new v("Invalid " + V + " `" + te + "` of type " + ("`" + ae + "` supplied to `" + G + "`, expected a single ReactElement."));
        }
        return null;
      }
      return h($);
    }
    function m() {
      function $(D, B, G, V, te) {
        var re = D[B];
        if (!e.isValidElementType(re)) {
          var ae = q(re);
          return new v("Invalid " + V + " `" + te + "` of type " + ("`" + ae + "` supplied to `" + G + "`, expected a single ReactElement type."));
        }
        return null;
      }
      return h($);
    }
    function x($) {
      function D(B, G, V, te, re) {
        if (!(B[G] instanceof $)) {
          var ae = $.name || d, ne = z(B[G]);
          return new v("Invalid " + te + " `" + re + "` of type " + ("`" + ne + "` supplied to `" + V + "`, expected ") + ("instance of `" + ae + "`."));
        }
        return null;
      }
      return h(D);
    }
    function _($) {
      if (!Array.isArray($))
        return process.env.NODE_ENV !== "production" && (arguments.length > 1 ? a(
          "Invalid arguments supplied to oneOf, expected an array, got " + arguments.length + " arguments. A common mistake is to write oneOf(x, y, z) instead of oneOf([x, y, z])."
        ) : a("Invalid argument supplied to oneOf, expected an array.")), o;
      function D(B, G, V, te, re) {
        for (var ae = B[G], ne = 0; ne < $.length; ne++)
          if (y(ae, $[ne]))
            return null;
        var F = JSON.stringify($, function(ee, C) {
          var se = L(C);
          return se === "symbol" ? String(C) : C;
        });
        return new v("Invalid " + te + " `" + re + "` of value `" + String(ae) + "` " + ("supplied to `" + V + "`, expected one of " + F + "."));
      }
      return h(D);
    }
    function P($) {
      function D(B, G, V, te, re) {
        if (typeof $ != "function")
          return new v("Property `" + re + "` of component `" + V + "` has invalid PropType notation inside objectOf.");
        var ae = B[G], ne = q(ae);
        if (ne !== "object")
          return new v("Invalid " + te + " `" + re + "` of type " + ("`" + ne + "` supplied to `" + V + "`, expected an object."));
        for (var F in ae)
          if (n(ae, F)) {
            var H = $(ae, F, V, te, re + "." + F, r);
            if (H instanceof Error)
              return H;
          }
        return null;
      }
      return h(D);
    }
    function E($) {
      if (!Array.isArray($))
        return process.env.NODE_ENV !== "production" && a("Invalid argument supplied to oneOfType, expected an instance of array."), o;
      for (var D = 0; D < $.length; D++) {
        var B = $[D];
        if (typeof B != "function")
          return a(
            "Invalid argument supplied to oneOfType. Expected an array of check functions, but received " + U(B) + " at index " + D + "."
          ), o;
      }
      function G(V, te, re, ae, ne) {
        for (var F = [], H = 0; H < $.length; H++) {
          var ee = $[H], C = ee(V, te, re, ae, ne, r);
          if (C == null)
            return null;
          C.data && n(C.data, "expectedType") && F.push(C.data.expectedType);
        }
        var se = F.length > 0 ? ", expected one of type [" + F.join(", ") + "]" : "";
        return new v("Invalid " + ae + " `" + ne + "` supplied to " + ("`" + re + "`" + se + "."));
      }
      return h(G);
    }
    function I() {
      function $(D, B, G, V, te) {
        return R(D[B]) ? null : new v("Invalid " + V + " `" + te + "` supplied to " + ("`" + G + "`, expected a ReactNode."));
      }
      return h($);
    }
    function S($, D, B, G, V) {
      return new v(
        ($ || "React class") + ": " + D + " type `" + B + "." + G + "` is invalid; it must be a function, usually from the `prop-types` package, but received `" + V + "`."
      );
    }
    function j($) {
      function D(B, G, V, te, re) {
        var ae = B[G], ne = q(ae);
        if (ne !== "object")
          return new v("Invalid " + te + " `" + re + "` of type `" + ne + "` " + ("supplied to `" + V + "`, expected `object`."));
        for (var F in $) {
          var H = $[F];
          if (typeof H != "function")
            return S(V, te, re, F, L(H));
          var ee = H(ae, F, V, te, re + "." + F, r);
          if (ee)
            return ee;
        }
        return null;
      }
      return h(D);
    }
    function M($) {
      function D(B, G, V, te, re) {
        var ae = B[G], ne = q(ae);
        if (ne !== "object")
          return new v("Invalid " + te + " `" + re + "` of type `" + ne + "` " + ("supplied to `" + V + "`, expected `object`."));
        var F = t({}, B[G], $);
        for (var H in F) {
          var ee = $[H];
          if (n($, H) && typeof ee != "function")
            return S(V, te, re, H, L(ee));
          if (!ee)
            return new v(
              "Invalid " + te + " `" + re + "` key `" + H + "` supplied to `" + V + "`.\nBad object: " + JSON.stringify(B[G], null, "  ") + `
Valid keys: ` + JSON.stringify(Object.keys($), null, "  ")
            );
          var C = ee(ae, H, V, te, re + "." + H, r);
          if (C)
            return C;
        }
        return null;
      }
      return h(D);
    }
    function R($) {
      switch (typeof $) {
        case "number":
        case "string":
        case "undefined":
          return !0;
        case "boolean":
          return !$;
        case "object":
          if (Array.isArray($))
            return $.every(R);
          if ($ === null || u($))
            return !0;
          var D = l($);
          if (D) {
            var B = D.call($), G;
            if (D !== $.entries) {
              for (; !(G = B.next()).done; )
                if (!R(G.value))
                  return !1;
            } else
              for (; !(G = B.next()).done; ) {
                var V = G.value;
                if (V && !R(V[1]))
                  return !1;
              }
          } else
            return !1;
          return !0;
        default:
          return !1;
      }
    }
    function k($, D) {
      return $ === "symbol" ? !0 : D ? D["@@toStringTag"] === "Symbol" || typeof Symbol == "function" && D instanceof Symbol : !1;
    }
    function q($) {
      var D = typeof $;
      return Array.isArray($) ? "array" : $ instanceof RegExp ? "object" : k(D, $) ? "symbol" : D;
    }
    function L($) {
      if (typeof $ > "u" || $ === null)
        return "" + $;
      var D = q($);
      if (D === "object") {
        if ($ instanceof Date)
          return "date";
        if ($ instanceof RegExp)
          return "regexp";
      }
      return D;
    }
    function U($) {
      var D = L($);
      switch (D) {
        case "array":
        case "object":
          return "an " + D;
        case "boolean":
        case "date":
        case "regexp":
          return "a " + D;
        default:
          return D;
      }
    }
    function z($) {
      return !$.constructor || !$.constructor.name ? d : $.constructor.name;
    }
    return p.checkPropTypes = i, p.resetWarningCache = i.resetWarningCache, p.PropTypes = p, p;
  }, Qc;
}
var el, _g;
function uI() {
  if (_g) return el;
  _g = 1;
  var e = /* @__PURE__ */ Wd();
  function t() {
  }
  function r() {
  }
  return r.resetWarningCache = t, el = function() {
    function n(o, u, s, c, f, l) {
      if (l !== e) {
        var d = new Error(
          "Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types"
        );
        throw d.name = "Invariant Violation", d;
      }
    }
    n.isRequired = n;
    function i() {
      return n;
    }
    var a = {
      array: n,
      bigint: n,
      bool: n,
      func: n,
      number: n,
      object: n,
      string: n,
      symbol: n,
      any: n,
      arrayOf: i,
      element: n,
      elementType: n,
      instanceOf: i,
      node: n,
      objectOf: i,
      oneOf: i,
      oneOfType: i,
      shape: i,
      exact: i,
      checkPropTypes: r,
      resetWarningCache: t
    };
    return a.PropTypes = a, a;
  }, el;
}
var Sg;
function sI() {
  if (Sg) return Li.exports;
  if (Sg = 1, process.env.NODE_ENV !== "production") {
    var e = gw(), t = !0;
    Li.exports = /* @__PURE__ */ oI()(e.isElement, t);
  } else
    Li.exports = /* @__PURE__ */ uI()();
  return Li.exports;
}
var cI = /* @__PURE__ */ sI();
const be = /* @__PURE__ */ Pe(cI);
var lI = Object.getOwnPropertyNames, fI = Object.getOwnPropertySymbols, dI = Object.prototype.hasOwnProperty;
function Pg(e, t) {
  return function(n, i, a) {
    return e(n, i, a) && t(n, i, a);
  };
}
function Fi(e) {
  return function(r, n, i) {
    if (!r || !n || typeof r != "object" || typeof n != "object")
      return e(r, n, i);
    var a = i.cache, o = a.get(r), u = a.get(n);
    if (o && u)
      return o === n && u === r;
    a.set(r, n), a.set(n, r);
    var s = e(r, n, i);
    return a.delete(r), a.delete(n), s;
  };
}
function Ag(e) {
  return lI(e).concat(fI(e));
}
var pI = Object.hasOwn || function(e, t) {
  return dI.call(e, t);
};
function wr(e, t) {
  return e === t || !e && !t && e !== e && t !== t;
}
var hI = "__v", vI = "__o", yI = "_owner", Eg = Object.getOwnPropertyDescriptor, Tg = Object.keys;
function mI(e, t, r) {
  var n = e.length;
  if (t.length !== n)
    return !1;
  for (; n-- > 0; )
    if (!r.equals(e[n], t[n], n, n, e, t, r))
      return !1;
  return !0;
}
function gI(e, t) {
  return wr(e.getTime(), t.getTime());
}
function bI(e, t) {
  return e.name === t.name && e.message === t.message && e.cause === t.cause && e.stack === t.stack;
}
function xI(e, t) {
  return e === t;
}
function jg(e, t, r) {
  var n = e.size;
  if (n !== t.size)
    return !1;
  if (!n)
    return !0;
  for (var i = new Array(n), a = e.entries(), o, u, s = 0; (o = a.next()) && !o.done; ) {
    for (var c = t.entries(), f = !1, l = 0; (u = c.next()) && !u.done; ) {
      if (i[l]) {
        l++;
        continue;
      }
      var d = o.value, p = u.value;
      if (r.equals(d[0], p[0], s, l, e, t, r) && r.equals(d[1], p[1], d[0], p[0], e, t, r)) {
        f = i[l] = !0;
        break;
      }
      l++;
    }
    if (!f)
      return !1;
    s++;
  }
  return !0;
}
var wI = wr;
function OI(e, t, r) {
  var n = Tg(e), i = n.length;
  if (Tg(t).length !== i)
    return !1;
  for (; i-- > 0; )
    if (!xw(e, t, r, n[i]))
      return !1;
  return !0;
}
function An(e, t, r) {
  var n = Ag(e), i = n.length;
  if (Ag(t).length !== i)
    return !1;
  for (var a, o, u; i-- > 0; )
    if (a = n[i], !xw(e, t, r, a) || (o = Eg(e, a), u = Eg(t, a), (o || u) && (!o || !u || o.configurable !== u.configurable || o.enumerable !== u.enumerable || o.writable !== u.writable)))
      return !1;
  return !0;
}
function _I(e, t) {
  return wr(e.valueOf(), t.valueOf());
}
function SI(e, t) {
  return e.source === t.source && e.flags === t.flags;
}
function Cg(e, t, r) {
  var n = e.size;
  if (n !== t.size)
    return !1;
  if (!n)
    return !0;
  for (var i = new Array(n), a = e.values(), o, u; (o = a.next()) && !o.done; ) {
    for (var s = t.values(), c = !1, f = 0; (u = s.next()) && !u.done; ) {
      if (!i[f] && r.equals(o.value, u.value, o.value, u.value, e, t, r)) {
        c = i[f] = !0;
        break;
      }
      f++;
    }
    if (!c)
      return !1;
  }
  return !0;
}
function PI(e, t) {
  var r = e.length;
  if (t.length !== r)
    return !1;
  for (; r-- > 0; )
    if (e[r] !== t[r])
      return !1;
  return !0;
}
function AI(e, t) {
  return e.hostname === t.hostname && e.pathname === t.pathname && e.protocol === t.protocol && e.port === t.port && e.hash === t.hash && e.username === t.username && e.password === t.password;
}
function xw(e, t, r, n) {
  return (n === yI || n === vI || n === hI) && (e.$$typeof || t.$$typeof) ? !0 : pI(t, n) && r.equals(e[n], t[n], n, n, e, t, r);
}
var EI = "[object Arguments]", TI = "[object Boolean]", jI = "[object Date]", CI = "[object Error]", MI = "[object Map]", II = "[object Number]", $I = "[object Object]", RI = "[object RegExp]", kI = "[object Set]", NI = "[object String]", DI = "[object URL]", qI = Array.isArray, Mg = typeof ArrayBuffer == "function" && ArrayBuffer.isView ? ArrayBuffer.isView : null, Ig = Object.assign, LI = Object.prototype.toString.call.bind(Object.prototype.toString);
function BI(e) {
  var t = e.areArraysEqual, r = e.areDatesEqual, n = e.areErrorsEqual, i = e.areFunctionsEqual, a = e.areMapsEqual, o = e.areNumbersEqual, u = e.areObjectsEqual, s = e.arePrimitiveWrappersEqual, c = e.areRegExpsEqual, f = e.areSetsEqual, l = e.areTypedArraysEqual, d = e.areUrlsEqual;
  return function(y, v, h) {
    if (y === v)
      return !0;
    if (y == null || v == null)
      return !1;
    var g = typeof y;
    if (g !== typeof v)
      return !1;
    if (g !== "object")
      return g === "number" ? o(y, v, h) : g === "function" ? i(y, v, h) : !1;
    var w = y.constructor;
    if (w !== v.constructor)
      return !1;
    if (w === Object)
      return u(y, v, h);
    if (qI(y))
      return t(y, v, h);
    if (Mg != null && Mg(y))
      return l(y, v, h);
    if (w === Date)
      return r(y, v, h);
    if (w === RegExp)
      return c(y, v, h);
    if (w === Map)
      return a(y, v, h);
    if (w === Set)
      return f(y, v, h);
    var b = LI(y);
    return b === jI ? r(y, v, h) : b === RI ? c(y, v, h) : b === MI ? a(y, v, h) : b === kI ? f(y, v, h) : b === $I ? typeof y.then != "function" && typeof v.then != "function" && u(y, v, h) : b === DI ? d(y, v, h) : b === CI ? n(y, v, h) : b === EI ? u(y, v, h) : b === TI || b === II || b === NI ? s(y, v, h) : !1;
  };
}
function FI(e) {
  var t = e.circular, r = e.createCustomConfig, n = e.strict, i = {
    areArraysEqual: n ? An : mI,
    areDatesEqual: gI,
    areErrorsEqual: bI,
    areFunctionsEqual: xI,
    areMapsEqual: n ? Pg(jg, An) : jg,
    areNumbersEqual: wI,
    areObjectsEqual: n ? An : OI,
    arePrimitiveWrappersEqual: _I,
    areRegExpsEqual: SI,
    areSetsEqual: n ? Pg(Cg, An) : Cg,
    areTypedArraysEqual: n ? An : PI,
    areUrlsEqual: AI
  };
  if (r && (i = Ig({}, i, r(i))), t) {
    var a = Fi(i.areArraysEqual), o = Fi(i.areMapsEqual), u = Fi(i.areObjectsEqual), s = Fi(i.areSetsEqual);
    i = Ig({}, i, {
      areArraysEqual: a,
      areMapsEqual: o,
      areObjectsEqual: u,
      areSetsEqual: s
    });
  }
  return i;
}
function zI(e) {
  return function(t, r, n, i, a, o, u) {
    return e(t, r, u);
  };
}
function UI(e) {
  var t = e.circular, r = e.comparator, n = e.createState, i = e.equals, a = e.strict;
  if (n)
    return function(s, c) {
      var f = n(), l = f.cache, d = l === void 0 ? t ? /* @__PURE__ */ new WeakMap() : void 0 : l, p = f.meta;
      return r(s, c, {
        cache: d,
        equals: i,
        meta: p,
        strict: a
      });
    };
  if (t)
    return function(s, c) {
      return r(s, c, {
        cache: /* @__PURE__ */ new WeakMap(),
        equals: i,
        meta: void 0,
        strict: a
      });
    };
  var o = {
    cache: void 0,
    equals: i,
    meta: void 0,
    strict: a
  };
  return function(s, c) {
    return r(s, c, o);
  };
}
var WI = tr();
tr({ strict: !0 });
tr({ circular: !0 });
tr({
  circular: !0,
  strict: !0
});
tr({
  createInternalComparator: function() {
    return wr;
  }
});
tr({
  strict: !0,
  createInternalComparator: function() {
    return wr;
  }
});
tr({
  circular: !0,
  createInternalComparator: function() {
    return wr;
  }
});
tr({
  circular: !0,
  createInternalComparator: function() {
    return wr;
  },
  strict: !0
});
function tr(e) {
  e === void 0 && (e = {});
  var t = e.circular, r = t === void 0 ? !1 : t, n = e.createInternalComparator, i = e.createState, a = e.strict, o = a === void 0 ? !1 : a, u = FI(e), s = BI(u), c = n ? n(s) : zI(s);
  return UI({ circular: r, comparator: s, createState: i, equals: c, strict: o });
}
function GI(e) {
  typeof requestAnimationFrame < "u" && requestAnimationFrame(e);
}
function $g(e) {
  var t = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0, r = -1, n = function i(a) {
    r < 0 && (r = a), a - r > t ? (e(a), r = -1) : GI(i);
  };
  requestAnimationFrame(n);
}
function hf(e) {
  "@babel/helpers - typeof";
  return hf = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, hf(e);
}
function HI(e) {
  return XI(e) || YI(e) || VI(e) || KI();
}
function KI() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function VI(e, t) {
  if (e) {
    if (typeof e == "string") return Rg(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Rg(e, t);
  }
}
function Rg(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function YI(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function XI(e) {
  if (Array.isArray(e)) return e;
}
function ZI() {
  var e = {}, t = function() {
    return null;
  }, r = !1, n = function i(a) {
    if (!r) {
      if (Array.isArray(a)) {
        if (!a.length)
          return;
        var o = a, u = HI(o), s = u[0], c = u.slice(1);
        if (typeof s == "number") {
          $g(i.bind(null, c), s);
          return;
        }
        i(s), $g(i.bind(null, c));
        return;
      }
      hf(a) === "object" && (e = a, t(e)), typeof a == "function" && a();
    }
  };
  return {
    stop: function() {
      r = !0;
    },
    start: function(a) {
      r = !1, n(a);
    },
    subscribe: function(a) {
      return t = a, function() {
        t = function() {
          return null;
        };
      };
    }
  };
}
function ni(e) {
  "@babel/helpers - typeof";
  return ni = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ni(e);
}
function kg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Ng(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? kg(Object(r), !0).forEach(function(n) {
      ww(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : kg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function ww(e, t, r) {
  return t = JI(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function JI(e) {
  var t = QI(e, "string");
  return ni(t) === "symbol" ? t : String(t);
}
function QI(e, t) {
  if (ni(e) !== "object" || e === null) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ni(n) !== "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var e$ = function(t, r) {
  return [Object.keys(t), Object.keys(r)].reduce(function(n, i) {
    return n.filter(function(a) {
      return i.includes(a);
    });
  });
}, t$ = function(t) {
  return t;
}, r$ = function(t) {
  return t.replace(/([A-Z])/g, function(r) {
    return "-".concat(r.toLowerCase());
  });
}, Rn = function(t, r) {
  return Object.keys(r).reduce(function(n, i) {
    return Ng(Ng({}, n), {}, ww({}, i, t(i, r[i])));
  }, {});
}, Dg = function(t, r, n) {
  return t.map(function(i) {
    return "".concat(r$(i), " ").concat(r, "ms ").concat(n);
  }).join(",");
}, n$ = process.env.NODE_ENV !== "production", xa = function(t, r, n, i, a, o, u, s) {
  if (n$ && typeof console < "u" && console.warn && (r === void 0 && console.warn("LogUtils requires an error message argument"), !t))
    if (r === void 0)
      console.warn("Minified exception occurred; use the non-minified dev environment for the full error message and additional helpful warnings.");
    else {
      var c = [n, i, a, o, u, s], f = 0;
      console.warn(r.replace(/%s/g, function() {
        return c[f++];
      }));
    }
};
function i$(e, t) {
  return u$(e) || o$(e, t) || Ow(e, t) || a$();
}
function a$() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function o$(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function u$(e) {
  if (Array.isArray(e)) return e;
}
function s$(e) {
  return f$(e) || l$(e) || Ow(e) || c$();
}
function c$() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Ow(e, t) {
  if (e) {
    if (typeof e == "string") return vf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return vf(e, t);
  }
}
function l$(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function f$(e) {
  if (Array.isArray(e)) return vf(e);
}
function vf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
var wa = 1e-4, _w = function(t, r) {
  return [0, 3 * t, 3 * r - 6 * t, 3 * t - 3 * r + 1];
}, Sw = function(t, r) {
  return t.map(function(n, i) {
    return n * Math.pow(r, i);
  }).reduce(function(n, i) {
    return n + i;
  });
}, qg = function(t, r) {
  return function(n) {
    var i = _w(t, r);
    return Sw(i, n);
  };
}, d$ = function(t, r) {
  return function(n) {
    var i = _w(t, r), a = [].concat(s$(i.map(function(o, u) {
      return o * u;
    }).slice(1)), [0]);
    return Sw(a, n);
  };
}, Lg = function() {
  for (var t = arguments.length, r = new Array(t), n = 0; n < t; n++)
    r[n] = arguments[n];
  var i = r[0], a = r[1], o = r[2], u = r[3];
  if (r.length === 1)
    switch (r[0]) {
      case "linear":
        i = 0, a = 0, o = 1, u = 1;
        break;
      case "ease":
        i = 0.25, a = 0.1, o = 0.25, u = 1;
        break;
      case "ease-in":
        i = 0.42, a = 0, o = 1, u = 1;
        break;
      case "ease-out":
        i = 0.42, a = 0, o = 0.58, u = 1;
        break;
      case "ease-in-out":
        i = 0, a = 0, o = 0.58, u = 1;
        break;
      default: {
        var s = r[0].split("(");
        if (s[0] === "cubic-bezier" && s[1].split(")")[0].split(",").length === 4) {
          var c = s[1].split(")")[0].split(",").map(function(h) {
            return parseFloat(h);
          }), f = i$(c, 4);
          i = f[0], a = f[1], o = f[2], u = f[3];
        } else
          xa(!1, "[configBezier]: arguments should be one of oneOf 'linear', 'ease', 'ease-in', 'ease-out', 'ease-in-out','cubic-bezier(x1,y1,x2,y2)', instead received %s", r);
      }
    }
  xa([i, o, a, u].every(function(h) {
    return typeof h == "number" && h >= 0 && h <= 1;
  }), "[configBezier]: arguments should be x1, y1, x2, y2 of [0, 1] instead received %s", r);
  var l = qg(i, o), d = qg(a, u), p = d$(i, o), y = function(g) {
    return g > 1 ? 1 : g < 0 ? 0 : g;
  }, v = function(g) {
    for (var w = g > 1 ? 1 : g, b = w, O = 0; O < 8; ++O) {
      var m = l(b) - w, x = p(b);
      if (Math.abs(m - w) < wa || x < wa)
        return d(b);
      b = y(b - m / x);
    }
    return d(b);
  };
  return v.isStepper = !1, v;
}, p$ = function() {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, r = t.stiff, n = r === void 0 ? 100 : r, i = t.damping, a = i === void 0 ? 8 : i, o = t.dt, u = o === void 0 ? 17 : o, s = function(f, l, d) {
    var p = -(f - l) * n, y = d * a, v = d + (p - y) * u / 1e3, h = d * u / 1e3 + f;
    return Math.abs(h - l) < wa && Math.abs(v) < wa ? [l, 0] : [h, v];
  };
  return s.isStepper = !0, s.dt = u, s;
}, h$ = function() {
  for (var t = arguments.length, r = new Array(t), n = 0; n < t; n++)
    r[n] = arguments[n];
  var i = r[0];
  if (typeof i == "string")
    switch (i) {
      case "ease":
      case "ease-in-out":
      case "ease-out":
      case "ease-in":
      case "linear":
        return Lg(i);
      case "spring":
        return p$();
      default:
        if (i.split("(")[0] === "cubic-bezier")
          return Lg(i);
        xa(!1, "[configEasing]: first argument should be one of 'ease', 'ease-in', 'ease-out', 'ease-in-out','cubic-bezier(x1,y1,x2,y2)', 'linear' and 'spring', instead  received %s", r);
    }
  return typeof i == "function" ? i : (xa(!1, "[configEasing]: first argument type should be function or string, instead received %s", r), null);
};
function ii(e) {
  "@babel/helpers - typeof";
  return ii = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ii(e);
}
function Bg(e) {
  return m$(e) || y$(e) || Pw(e) || v$();
}
function v$() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function y$(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function m$(e) {
  if (Array.isArray(e)) return mf(e);
}
function Fg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Be(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Fg(Object(r), !0).forEach(function(n) {
      yf(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Fg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function yf(e, t, r) {
  return t = g$(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function g$(e) {
  var t = b$(e, "string");
  return ii(t) === "symbol" ? t : String(t);
}
function b$(e, t) {
  if (ii(e) !== "object" || e === null) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ii(n) !== "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function x$(e, t) {
  return _$(e) || O$(e, t) || Pw(e, t) || w$();
}
function w$() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function Pw(e, t) {
  if (e) {
    if (typeof e == "string") return mf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return mf(e, t);
  }
}
function mf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function O$(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function _$(e) {
  if (Array.isArray(e)) return e;
}
var Oa = function(t, r, n) {
  return t + (r - t) * n;
}, gf = function(t) {
  var r = t.from, n = t.to;
  return r !== n;
}, S$ = function e(t, r, n) {
  var i = Rn(function(a, o) {
    if (gf(o)) {
      var u = t(o.from, o.to, o.velocity), s = x$(u, 2), c = s[0], f = s[1];
      return Be(Be({}, o), {}, {
        from: c,
        velocity: f
      });
    }
    return o;
  }, r);
  return n < 1 ? Rn(function(a, o) {
    return gf(o) ? Be(Be({}, o), {}, {
      velocity: Oa(o.velocity, i[a].velocity, n),
      from: Oa(o.from, i[a].from, n)
    }) : o;
  }, r) : e(t, i, n - 1);
};
const P$ = function(e, t, r, n, i) {
  var a = e$(e, t), o = a.reduce(function(h, g) {
    return Be(Be({}, h), {}, yf({}, g, [e[g], t[g]]));
  }, {}), u = a.reduce(function(h, g) {
    return Be(Be({}, h), {}, yf({}, g, {
      from: e[g],
      velocity: 0,
      to: t[g]
    }));
  }, {}), s = -1, c, f, l = function() {
    return null;
  }, d = function() {
    return Rn(function(g, w) {
      return w.from;
    }, u);
  }, p = function() {
    return !Object.values(u).filter(gf).length;
  }, y = function(g) {
    c || (c = g);
    var w = g - c, b = w / r.dt;
    u = S$(r, u, b), i(Be(Be(Be({}, e), t), d())), c = g, p() || (s = requestAnimationFrame(l));
  }, v = function(g) {
    f || (f = g);
    var w = (g - f) / n, b = Rn(function(m, x) {
      return Oa.apply(void 0, Bg(x).concat([r(w)]));
    }, o);
    if (i(Be(Be(Be({}, e), t), b)), w < 1)
      s = requestAnimationFrame(l);
    else {
      var O = Rn(function(m, x) {
        return Oa.apply(void 0, Bg(x).concat([r(1)]));
      }, o);
      i(Be(Be(Be({}, e), t), O));
    }
  };
  return l = r.isStepper ? y : v, function() {
    return requestAnimationFrame(l), function() {
      cancelAnimationFrame(s);
    };
  };
};
function Gr(e) {
  "@babel/helpers - typeof";
  return Gr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Gr(e);
}
var A$ = ["children", "begin", "duration", "attributeName", "easing", "isActive", "steps", "from", "to", "canBegin", "onAnimationEnd", "shouldReAnimate", "onAnimationReStart"];
function E$(e, t) {
  if (e == null) return {};
  var r = T$(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function T$(e, t) {
  if (e == null) return {};
  var r = {}, n = Object.keys(e), i, a;
  for (a = 0; a < n.length; a++)
    i = n[a], !(t.indexOf(i) >= 0) && (r[i] = e[i]);
  return r;
}
function tl(e) {
  return I$(e) || M$(e) || C$(e) || j$();
}
function j$() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function C$(e, t) {
  if (e) {
    if (typeof e == "string") return bf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return bf(e, t);
  }
}
function M$(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function I$(e) {
  if (Array.isArray(e)) return bf(e);
}
function bf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function zg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function ft(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? zg(Object(r), !0).forEach(function(n) {
      Cn(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : zg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Cn(e, t, r) {
  return t = Aw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function $$(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function R$(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Aw(n.key), n);
  }
}
function k$(e, t, r) {
  return R$(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function Aw(e) {
  var t = N$(e, "string");
  return Gr(t) === "symbol" ? t : String(t);
}
function N$(e, t) {
  if (Gr(e) !== "object" || e === null) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Gr(n) !== "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function D$(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && xf(e, t);
}
function xf(e, t) {
  return xf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, xf(e, t);
}
function q$(e) {
  var t = L$();
  return function() {
    var n = _a(e), i;
    if (t) {
      var a = _a(this).constructor;
      i = Reflect.construct(n, arguments, a);
    } else
      i = n.apply(this, arguments);
    return wf(this, i);
  };
}
function wf(e, t) {
  if (t && (Gr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return Of(e);
}
function Of(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function L$() {
  if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
  if (typeof Proxy == "function") return !0;
  try {
    return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    })), !0;
  } catch {
    return !1;
  }
}
function _a(e) {
  return _a = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, _a(e);
}
var Dt = /* @__PURE__ */ function(e) {
  D$(r, e);
  var t = q$(r);
  function r(n, i) {
    var a;
    $$(this, r), a = t.call(this, n, i);
    var o = a.props, u = o.isActive, s = o.attributeName, c = o.from, f = o.to, l = o.steps, d = o.children, p = o.duration;
    if (a.handleStyleChange = a.handleStyleChange.bind(Of(a)), a.changeStyle = a.changeStyle.bind(Of(a)), !u || p <= 0)
      return a.state = {
        style: {}
      }, typeof d == "function" && (a.state = {
        style: f
      }), wf(a);
    if (l && l.length)
      a.state = {
        style: l[0].style
      };
    else if (c) {
      if (typeof d == "function")
        return a.state = {
          style: c
        }, wf(a);
      a.state = {
        style: s ? Cn({}, s, c) : c
      };
    } else
      a.state = {
        style: {}
      };
    return a;
  }
  return k$(r, [{
    key: "componentDidMount",
    value: function() {
      var i = this.props, a = i.isActive, o = i.canBegin;
      this.mounted = !0, !(!a || !o) && this.runAnimation(this.props);
    }
  }, {
    key: "componentDidUpdate",
    value: function(i) {
      var a = this.props, o = a.isActive, u = a.canBegin, s = a.attributeName, c = a.shouldReAnimate, f = a.to, l = a.from, d = this.state.style;
      if (u) {
        if (!o) {
          var p = {
            style: s ? Cn({}, s, f) : f
          };
          this.state && d && (s && d[s] !== f || !s && d !== f) && this.setState(p);
          return;
        }
        if (!(WI(i.to, f) && i.canBegin && i.isActive)) {
          var y = !i.canBegin || !i.isActive;
          this.manager && this.manager.stop(), this.stopJSAnimation && this.stopJSAnimation();
          var v = y || c ? l : i.to;
          if (this.state && d) {
            var h = {
              style: s ? Cn({}, s, v) : v
            };
            (s && d[s] !== v || !s && d !== v) && this.setState(h);
          }
          this.runAnimation(ft(ft({}, this.props), {}, {
            from: v,
            begin: 0
          }));
        }
      }
    }
  }, {
    key: "componentWillUnmount",
    value: function() {
      this.mounted = !1;
      var i = this.props.onAnimationEnd;
      this.unSubscribe && this.unSubscribe(), this.manager && (this.manager.stop(), this.manager = null), this.stopJSAnimation && this.stopJSAnimation(), i && i();
    }
  }, {
    key: "handleStyleChange",
    value: function(i) {
      this.changeStyle(i);
    }
  }, {
    key: "changeStyle",
    value: function(i) {
      this.mounted && this.setState({
        style: i
      });
    }
  }, {
    key: "runJSAnimation",
    value: function(i) {
      var a = this, o = i.from, u = i.to, s = i.duration, c = i.easing, f = i.begin, l = i.onAnimationEnd, d = i.onAnimationStart, p = P$(o, u, h$(c), s, this.changeStyle), y = function() {
        a.stopJSAnimation = p();
      };
      this.manager.start([d, f, y, s, l]);
    }
  }, {
    key: "runStepAnimation",
    value: function(i) {
      var a = this, o = i.steps, u = i.begin, s = i.onAnimationStart, c = o[0], f = c.style, l = c.duration, d = l === void 0 ? 0 : l, p = function(v, h, g) {
        if (g === 0)
          return v;
        var w = h.duration, b = h.easing, O = b === void 0 ? "ease" : b, m = h.style, x = h.properties, _ = h.onAnimationEnd, P = g > 0 ? o[g - 1] : h, E = x || Object.keys(m);
        if (typeof O == "function" || O === "spring")
          return [].concat(tl(v), [a.runJSAnimation.bind(a, {
            from: P.style,
            to: m,
            duration: w,
            easing: O
          }), w]);
        var I = Dg(E, w, O), S = ft(ft(ft({}, P.style), m), {}, {
          transition: I
        });
        return [].concat(tl(v), [S, w, _]).filter(t$);
      };
      return this.manager.start([s].concat(tl(o.reduce(p, [f, Math.max(d, u)])), [i.onAnimationEnd]));
    }
  }, {
    key: "runAnimation",
    value: function(i) {
      this.manager || (this.manager = ZI());
      var a = i.begin, o = i.duration, u = i.attributeName, s = i.to, c = i.easing, f = i.onAnimationStart, l = i.onAnimationEnd, d = i.steps, p = i.children, y = this.manager;
      if (this.unSubscribe = y.subscribe(this.handleStyleChange), typeof c == "function" || typeof p == "function" || c === "spring") {
        this.runJSAnimation(i);
        return;
      }
      if (d.length > 1) {
        this.runStepAnimation(i);
        return;
      }
      var v = u ? Cn({}, u, s) : s, h = Dg(Object.keys(v), o, c);
      y.start([f, a, ft(ft({}, v), {}, {
        transition: h
      }), o, l]);
    }
  }, {
    key: "render",
    value: function() {
      var i = this.props, a = i.children;
      i.begin;
      var o = i.duration;
      i.attributeName, i.easing;
      var u = i.isActive;
      i.steps, i.from, i.to, i.canBegin, i.onAnimationEnd, i.shouldReAnimate, i.onAnimationReStart;
      var s = E$(i, A$), c = fr.count(a), f = this.state.style;
      if (typeof a == "function")
        return a(f);
      if (!u || c === 0 || o <= 0)
        return a;
      var l = function(p) {
        var y = p.props, v = y.style, h = v === void 0 ? {} : v, g = y.className, w = /* @__PURE__ */ De(p, ft(ft({}, s), {}, {
          style: ft(ft({}, h), f),
          className: g
        }));
        return w;
      };
      return c === 1 ? l(fr.only(a)) : /* @__PURE__ */ T.createElement("div", null, fr.map(a, function(d) {
        return l(d);
      }));
    }
  }]), r;
}(Xt);
Dt.displayName = "Animate";
Dt.defaultProps = {
  begin: 0,
  duration: 1e3,
  from: "",
  to: "",
  attributeName: "",
  easing: "ease",
  isActive: !0,
  canBegin: !0,
  steps: [],
  onAnimationEnd: function() {
  },
  onAnimationStart: function() {
  }
};
Dt.propTypes = {
  from: be.oneOfType([be.object, be.string]),
  to: be.oneOfType([be.object, be.string]),
  attributeName: be.string,
  // animation duration
  duration: be.number,
  begin: be.number,
  easing: be.oneOfType([be.string, be.func]),
  steps: be.arrayOf(be.shape({
    duration: be.number.isRequired,
    style: be.object.isRequired,
    easing: be.oneOfType([be.oneOf(["ease", "ease-in", "ease-out", "ease-in-out", "linear"]), be.func]),
    // transition css properties(dash case), optional
    properties: be.arrayOf("string"),
    onAnimationEnd: be.func
  })),
  children: be.oneOfType([be.node, be.func]),
  isActive: be.bool,
  canBegin: be.bool,
  onAnimationEnd: be.func,
  // decide if it should reanimate with initial from style when props change
  shouldReAnimate: be.bool,
  onAnimationStart: be.func,
  onAnimationReStart: be.func
};
function ai(e) {
  "@babel/helpers - typeof";
  return ai = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ai(e);
}
function Sa() {
  return Sa = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Sa.apply(this, arguments);
}
function B$(e, t) {
  return W$(e) || U$(e, t) || z$(e, t) || F$();
}
function F$() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function z$(e, t) {
  if (e) {
    if (typeof e == "string") return Ug(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Ug(e, t);
  }
}
function Ug(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function U$(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function W$(e) {
  if (Array.isArray(e)) return e;
}
function Wg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Gg(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Wg(Object(r), !0).forEach(function(n) {
      G$(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Wg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function G$(e, t, r) {
  return t = H$(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function H$(e) {
  var t = K$(e, "string");
  return ai(t) == "symbol" ? t : t + "";
}
function K$(e, t) {
  if (ai(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ai(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Hg = function(t, r, n, i, a) {
  var o = Math.min(Math.abs(n) / 2, Math.abs(i) / 2), u = i >= 0 ? 1 : -1, s = n >= 0 ? 1 : -1, c = i >= 0 && n >= 0 || i < 0 && n < 0 ? 1 : 0, f;
  if (o > 0 && a instanceof Array) {
    for (var l = [0, 0, 0, 0], d = 0, p = 4; d < p; d++)
      l[d] = a[d] > o ? o : a[d];
    f = "M".concat(t, ",").concat(r + u * l[0]), l[0] > 0 && (f += "A ".concat(l[0], ",").concat(l[0], ",0,0,").concat(c, ",").concat(t + s * l[0], ",").concat(r)), f += "L ".concat(t + n - s * l[1], ",").concat(r), l[1] > 0 && (f += "A ".concat(l[1], ",").concat(l[1], ",0,0,").concat(c, `,
        `).concat(t + n, ",").concat(r + u * l[1])), f += "L ".concat(t + n, ",").concat(r + i - u * l[2]), l[2] > 0 && (f += "A ".concat(l[2], ",").concat(l[2], ",0,0,").concat(c, `,
        `).concat(t + n - s * l[2], ",").concat(r + i)), f += "L ".concat(t + s * l[3], ",").concat(r + i), l[3] > 0 && (f += "A ".concat(l[3], ",").concat(l[3], ",0,0,").concat(c, `,
        `).concat(t, ",").concat(r + i - u * l[3])), f += "Z";
  } else if (o > 0 && a === +a && a > 0) {
    var y = Math.min(o, a);
    f = "M ".concat(t, ",").concat(r + u * y, `
            A `).concat(y, ",").concat(y, ",0,0,").concat(c, ",").concat(t + s * y, ",").concat(r, `
            L `).concat(t + n - s * y, ",").concat(r, `
            A `).concat(y, ",").concat(y, ",0,0,").concat(c, ",").concat(t + n, ",").concat(r + u * y, `
            L `).concat(t + n, ",").concat(r + i - u * y, `
            A `).concat(y, ",").concat(y, ",0,0,").concat(c, ",").concat(t + n - s * y, ",").concat(r + i, `
            L `).concat(t + s * y, ",").concat(r + i, `
            A `).concat(y, ",").concat(y, ",0,0,").concat(c, ",").concat(t, ",").concat(r + i - u * y, " Z");
  } else
    f = "M ".concat(t, ",").concat(r, " h ").concat(n, " v ").concat(i, " h ").concat(-n, " Z");
  return f;
}, V$ = function(t, r) {
  if (!t || !r)
    return !1;
  var n = t.x, i = t.y, a = r.x, o = r.y, u = r.width, s = r.height;
  if (Math.abs(u) > 0 && Math.abs(s) > 0) {
    var c = Math.min(a, a + u), f = Math.max(a, a + u), l = Math.min(o, o + s), d = Math.max(o, o + s);
    return n >= c && n <= f && i >= l && i <= d;
  }
  return !1;
}, Y$ = {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  // The radius of border
  // The radius of four corners when radius is a number
  // The radius of left-top, right-top, right-bottom, left-bottom when radius is an array
  radius: 0,
  isAnimationActive: !1,
  isUpdateAnimationActive: !1,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: "ease"
}, Gd = function(t) {
  var r = Gg(Gg({}, Y$), t), n = o0(), i = Er(-1), a = B$(i, 2), o = a[0], u = a[1];
  Xf(function() {
    if (n.current && n.current.getTotalLength)
      try {
        var O = n.current.getTotalLength();
        O && u(O);
      } catch {
      }
  }, []);
  var s = r.x, c = r.y, f = r.width, l = r.height, d = r.radius, p = r.className, y = r.animationEasing, v = r.animationDuration, h = r.animationBegin, g = r.isAnimationActive, w = r.isUpdateAnimationActive;
  if (s !== +s || c !== +c || f !== +f || l !== +l || f === 0 || l === 0)
    return null;
  var b = pe("recharts-rectangle", p);
  return w ? /* @__PURE__ */ T.createElement(Dt, {
    canBegin: o > 0,
    from: {
      width: f,
      height: l,
      x: s,
      y: c
    },
    to: {
      width: f,
      height: l,
      x: s,
      y: c
    },
    duration: v,
    animationEasing: y,
    isActive: w
  }, function(O) {
    var m = O.width, x = O.height, _ = O.x, P = O.y;
    return /* @__PURE__ */ T.createElement(Dt, {
      canBegin: o > 0,
      from: "0px ".concat(o === -1 ? 1 : o, "px"),
      to: "".concat(o, "px 0px"),
      attributeName: "strokeDasharray",
      begin: h,
      duration: v,
      isActive: g,
      easing: y
    }, /* @__PURE__ */ T.createElement("path", Sa({}, fe(r, !0), {
      className: b,
      d: Hg(_, P, m, x, d),
      ref: n
    })));
  }) : /* @__PURE__ */ T.createElement("path", Sa({}, fe(r, !0), {
    className: b,
    d: Hg(s, c, f, l, d)
  }));
};
function _f() {
  return _f = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, _f.apply(this, arguments);
}
var Hd = function(t) {
  var r = t.cx, n = t.cy, i = t.r, a = t.className, o = pe("recharts-dot", a);
  return r === +r && n === +n && i === +i ? /* @__PURE__ */ T.createElement("circle", _f({}, fe(t, !1), Gi(t), {
    className: o,
    cx: r,
    cy: n,
    r: i
  })) : null;
};
function oi(e) {
  "@babel/helpers - typeof";
  return oi = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, oi(e);
}
var X$ = ["x", "y", "top", "left", "width", "height", "className"];
function Sf() {
  return Sf = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Sf.apply(this, arguments);
}
function Kg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Z$(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Kg(Object(r), !0).forEach(function(n) {
      J$(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Kg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function J$(e, t, r) {
  return t = Q$(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Q$(e) {
  var t = eR(e, "string");
  return oi(t) == "symbol" ? t : t + "";
}
function eR(e, t) {
  if (oi(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (oi(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function tR(e, t) {
  if (e == null) return {};
  var r = rR(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function rR(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
var nR = function(t, r, n, i, a, o) {
  return "M".concat(t, ",").concat(a, "v").concat(i, "M").concat(o, ",").concat(r, "h").concat(n);
}, iR = function(t) {
  var r = t.x, n = r === void 0 ? 0 : r, i = t.y, a = i === void 0 ? 0 : i, o = t.top, u = o === void 0 ? 0 : o, s = t.left, c = s === void 0 ? 0 : s, f = t.width, l = f === void 0 ? 0 : f, d = t.height, p = d === void 0 ? 0 : d, y = t.className, v = tR(t, X$), h = Z$({
    x: n,
    y: a,
    top: u,
    left: c,
    width: l,
    height: p
  }, v);
  return !K(n) || !K(a) || !K(l) || !K(p) || !K(u) || !K(c) ? null : /* @__PURE__ */ T.createElement("path", Sf({}, fe(h, !0), {
    className: pe("recharts-cross", y),
    d: nR(n, a, l, p, u, c)
  }));
}, rl, Vg;
function aR() {
  if (Vg) return rl;
  Vg = 1;
  var e = G0(), t = e(Object.getPrototypeOf, Object);
  return rl = t, rl;
}
var nl, Yg;
function oR() {
  if (Yg) return nl;
  Yg = 1;
  var e = Lt(), t = aR(), r = Bt(), n = "[object Object]", i = Function.prototype, a = Object.prototype, o = i.toString, u = a.hasOwnProperty, s = o.call(Object);
  function c(f) {
    if (!r(f) || e(f) != n)
      return !1;
    var l = t(f);
    if (l === null)
      return !0;
    var d = u.call(l, "constructor") && l.constructor;
    return typeof d == "function" && d instanceof d && o.call(d) == s;
  }
  return nl = c, nl;
}
var uR = oR();
const sR = /* @__PURE__ */ Pe(uR);
var il, Xg;
function cR() {
  if (Xg) return il;
  Xg = 1;
  var e = Lt(), t = Bt(), r = "[object Boolean]";
  function n(i) {
    return i === !0 || i === !1 || t(i) && e(i) == r;
  }
  return il = n, il;
}
var lR = cR();
const fR = /* @__PURE__ */ Pe(lR);
function ui(e) {
  "@babel/helpers - typeof";
  return ui = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, ui(e);
}
function Pa() {
  return Pa = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Pa.apply(this, arguments);
}
function dR(e, t) {
  return yR(e) || vR(e, t) || hR(e, t) || pR();
}
function pR() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function hR(e, t) {
  if (e) {
    if (typeof e == "string") return Zg(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Zg(e, t);
  }
}
function Zg(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function vR(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function yR(e) {
  if (Array.isArray(e)) return e;
}
function Jg(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Qg(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Jg(Object(r), !0).forEach(function(n) {
      mR(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Jg(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function mR(e, t, r) {
  return t = gR(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function gR(e) {
  var t = bR(e, "string");
  return ui(t) == "symbol" ? t : t + "";
}
function bR(e, t) {
  if (ui(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (ui(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var eb = function(t, r, n, i, a) {
  var o = n - i, u;
  return u = "M ".concat(t, ",").concat(r), u += "L ".concat(t + n, ",").concat(r), u += "L ".concat(t + n - o / 2, ",").concat(r + a), u += "L ".concat(t + n - o / 2 - i, ",").concat(r + a), u += "L ".concat(t, ",").concat(r, " Z"), u;
}, xR = {
  x: 0,
  y: 0,
  upperWidth: 0,
  lowerWidth: 0,
  height: 0,
  isUpdateAnimationActive: !1,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: "ease"
}, wR = function(t) {
  var r = Qg(Qg({}, xR), t), n = o0(), i = Er(-1), a = dR(i, 2), o = a[0], u = a[1];
  Xf(function() {
    if (n.current && n.current.getTotalLength)
      try {
        var b = n.current.getTotalLength();
        b && u(b);
      } catch {
      }
  }, []);
  var s = r.x, c = r.y, f = r.upperWidth, l = r.lowerWidth, d = r.height, p = r.className, y = r.animationEasing, v = r.animationDuration, h = r.animationBegin, g = r.isUpdateAnimationActive;
  if (s !== +s || c !== +c || f !== +f || l !== +l || d !== +d || f === 0 && l === 0 || d === 0)
    return null;
  var w = pe("recharts-trapezoid", p);
  return g ? /* @__PURE__ */ T.createElement(Dt, {
    canBegin: o > 0,
    from: {
      upperWidth: 0,
      lowerWidth: 0,
      height: d,
      x: s,
      y: c
    },
    to: {
      upperWidth: f,
      lowerWidth: l,
      height: d,
      x: s,
      y: c
    },
    duration: v,
    animationEasing: y,
    isActive: g
  }, function(b) {
    var O = b.upperWidth, m = b.lowerWidth, x = b.height, _ = b.x, P = b.y;
    return /* @__PURE__ */ T.createElement(Dt, {
      canBegin: o > 0,
      from: "0px ".concat(o === -1 ? 1 : o, "px"),
      to: "".concat(o, "px 0px"),
      attributeName: "strokeDasharray",
      begin: h,
      duration: v,
      easing: y
    }, /* @__PURE__ */ T.createElement("path", Pa({}, fe(r, !0), {
      className: w,
      d: eb(_, P, O, m, x),
      ref: n
    })));
  }) : /* @__PURE__ */ T.createElement("g", null, /* @__PURE__ */ T.createElement("path", Pa({}, fe(r, !0), {
    className: w,
    d: eb(s, c, f, l, d)
  })));
}, OR = ["option", "shapeType", "propTransformer", "activeClassName", "isActive"];
function si(e) {
  "@babel/helpers - typeof";
  return si = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, si(e);
}
function _R(e, t) {
  if (e == null) return {};
  var r = SR(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function SR(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function tb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Aa(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? tb(Object(r), !0).forEach(function(n) {
      PR(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : tb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function PR(e, t, r) {
  return t = AR(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function AR(e) {
  var t = ER(e, "string");
  return si(t) == "symbol" ? t : t + "";
}
function ER(e, t) {
  if (si(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (si(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function TR(e, t) {
  return Aa(Aa({}, t), e);
}
function jR(e, t) {
  return e === "symbols";
}
function rb(e) {
  var t = e.shapeType, r = e.elementProps;
  switch (t) {
    case "rectangle":
      return /* @__PURE__ */ T.createElement(Gd, r);
    case "trapezoid":
      return /* @__PURE__ */ T.createElement(wR, r);
    case "sector":
      return /* @__PURE__ */ T.createElement(mw, r);
    case "symbols":
      if (jR(t))
        return /* @__PURE__ */ T.createElement(cd, r);
      break;
    default:
      return null;
  }
}
function CR(e) {
  return /* @__PURE__ */ xt(e) ? e.props : e;
}
function MR(e) {
  var t = e.option, r = e.shapeType, n = e.propTransformer, i = n === void 0 ? TR : n, a = e.activeClassName, o = a === void 0 ? "recharts-active-shape" : a, u = e.isActive, s = _R(e, OR), c;
  if (/* @__PURE__ */ xt(t))
    c = /* @__PURE__ */ De(t, Aa(Aa({}, s), CR(t)));
  else if (ue(t))
    c = t(s);
  else if (sR(t) && !fR(t)) {
    var f = i(t, s);
    c = /* @__PURE__ */ T.createElement(rb, {
      shapeType: r,
      elementProps: f
    });
  } else {
    var l = s;
    c = /* @__PURE__ */ T.createElement(rb, {
      shapeType: r,
      elementProps: l
    });
  }
  return u ? /* @__PURE__ */ T.createElement(je, {
    className: o
  }, c) : c;
}
function po(e, t) {
  return t != null && "trapezoids" in e.props;
}
function ho(e, t) {
  return t != null && "sectors" in e.props;
}
function ci(e, t) {
  return t != null && "points" in e.props;
}
function IR(e, t) {
  var r, n, i = e.x === (t == null || (r = t.labelViewBox) === null || r === void 0 ? void 0 : r.x) || e.x === t.x, a = e.y === (t == null || (n = t.labelViewBox) === null || n === void 0 ? void 0 : n.y) || e.y === t.y;
  return i && a;
}
function $R(e, t) {
  var r = e.endAngle === t.endAngle, n = e.startAngle === t.startAngle;
  return r && n;
}
function RR(e, t) {
  var r = e.x === t.x, n = e.y === t.y, i = e.z === t.z;
  return r && n && i;
}
function kR(e, t) {
  var r;
  return po(e, t) ? r = IR : ho(e, t) ? r = $R : ci(e, t) && (r = RR), r;
}
function NR(e, t) {
  var r;
  return po(e, t) ? r = "trapezoids" : ho(e, t) ? r = "sectors" : ci(e, t) && (r = "points"), r;
}
function DR(e, t) {
  if (po(e, t)) {
    var r;
    return (r = t.tooltipPayload) === null || r === void 0 || (r = r[0]) === null || r === void 0 || (r = r.payload) === null || r === void 0 ? void 0 : r.payload;
  }
  if (ho(e, t)) {
    var n;
    return (n = t.tooltipPayload) === null || n === void 0 || (n = n[0]) === null || n === void 0 || (n = n.payload) === null || n === void 0 ? void 0 : n.payload;
  }
  return ci(e, t) ? t.payload : {};
}
function qR(e) {
  var t = e.activeTooltipItem, r = e.graphicalItem, n = e.itemData, i = NR(r, t), a = DR(r, t), o = n.filter(function(s, c) {
    var f = co(a, s), l = r.props[i].filter(function(y) {
      var v = kR(r, t);
      return v(y, t);
    }), d = r.props[i].indexOf(l[l.length - 1]), p = c === d;
    return f && p;
  }), u = n.indexOf(o[o.length - 1]);
  return u;
}
var al, nb;
function LR() {
  if (nb) return al;
  nb = 1;
  var e = Math.ceil, t = Math.max;
  function r(n, i, a, o) {
    for (var u = -1, s = t(e((i - n) / (a || 1)), 0), c = Array(s); s--; )
      c[o ? s : ++u] = n, n += a;
    return c;
  }
  return al = r, al;
}
var ol, ib;
function Ew() {
  if (ib) return ol;
  ib = 1;
  var e = ux(), t = 1 / 0, r = 17976931348623157e292;
  function n(i) {
    if (!i)
      return i === 0 ? i : 0;
    if (i = e(i), i === t || i === -1 / 0) {
      var a = i < 0 ? -1 : 1;
      return a * r;
    }
    return i === i ? i : 0;
  }
  return ol = n, ol;
}
var ul, ab;
function BR() {
  if (ab) return ul;
  ab = 1;
  var e = LR(), t = eo(), r = Ew();
  function n(i) {
    return function(a, o, u) {
      return u && typeof u != "number" && t(a, o, u) && (o = u = void 0), a = r(a), o === void 0 ? (o = a, a = 0) : o = r(o), u = u === void 0 ? a < o ? 1 : -1 : r(u), e(a, o, u, i);
    };
  }
  return ul = n, ul;
}
var sl, ob;
function FR() {
  if (ob) return sl;
  ob = 1;
  var e = BR(), t = e();
  return sl = t, sl;
}
var zR = FR();
const Ea = /* @__PURE__ */ Pe(zR);
function li(e) {
  "@babel/helpers - typeof";
  return li = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, li(e);
}
function ub(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function sb(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? ub(Object(r), !0).forEach(function(n) {
      Tw(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ub(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Tw(e, t, r) {
  return t = UR(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function UR(e) {
  var t = WR(e, "string");
  return li(t) == "symbol" ? t : t + "";
}
function WR(e, t) {
  if (li(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (li(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var GR = ["Webkit", "Moz", "O", "ms"], HR = function(t, r) {
  var n = t.replace(/(\w)/, function(a) {
    return a.toUpperCase();
  }), i = GR.reduce(function(a, o) {
    return sb(sb({}, a), {}, Tw({}, o + n, r));
  }, {});
  return i[t] = r, i;
};
function Hr(e) {
  "@babel/helpers - typeof";
  return Hr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Hr(e);
}
function Ta() {
  return Ta = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Ta.apply(this, arguments);
}
function cb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function cl(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? cb(Object(r), !0).forEach(function(n) {
      Je(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : cb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function KR(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function lb(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Cw(n.key), n);
  }
}
function VR(e, t, r) {
  return lb(e.prototype, t), lb(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function YR(e, t, r) {
  return t = ja(t), XR(e, jw() ? Reflect.construct(t, r, ja(e).constructor) : t.apply(e, r));
}
function XR(e, t) {
  if (t && (Hr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return ZR(e);
}
function ZR(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function jw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (jw = function() {
    return !!e;
  })();
}
function ja(e) {
  return ja = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, ja(e);
}
function JR(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Pf(e, t);
}
function Pf(e, t) {
  return Pf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Pf(e, t);
}
function Je(e, t, r) {
  return t = Cw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Cw(e) {
  var t = QR(e, "string");
  return Hr(t) == "symbol" ? t : t + "";
}
function QR(e, t) {
  if (Hr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Hr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var ek = function(t) {
  var r = t.data, n = t.startIndex, i = t.endIndex, a = t.x, o = t.width, u = t.travellerWidth;
  if (!r || !r.length)
    return {};
  var s = r.length, c = In().domain(Ea(0, s)).range([a, a + o - u]), f = c.domain().map(function(l) {
    return c(l);
  });
  return {
    isTextActive: !1,
    isSlideMoving: !1,
    isTravellerMoving: !1,
    isTravellerFocused: !1,
    startX: c(n),
    endX: c(i),
    scale: c,
    scaleValues: f
  };
}, fb = function(t) {
  return t.changedTouches && !!t.changedTouches.length;
}, Kr = /* @__PURE__ */ function(e) {
  function t(r) {
    var n;
    return KR(this, t), n = YR(this, t, [r]), Je(n, "handleDrag", function(i) {
      n.leaveTimer && (clearTimeout(n.leaveTimer), n.leaveTimer = null), n.state.isTravellerMoving ? n.handleTravellerMove(i) : n.state.isSlideMoving && n.handleSlideDrag(i);
    }), Je(n, "handleTouchMove", function(i) {
      i.changedTouches != null && i.changedTouches.length > 0 && n.handleDrag(i.changedTouches[0]);
    }), Je(n, "handleDragEnd", function() {
      n.setState({
        isTravellerMoving: !1,
        isSlideMoving: !1
      }, function() {
        var i = n.props, a = i.endIndex, o = i.onDragEnd, u = i.startIndex;
        o == null || o({
          endIndex: a,
          startIndex: u
        });
      }), n.detachDragEndListener();
    }), Je(n, "handleLeaveWrapper", function() {
      (n.state.isTravellerMoving || n.state.isSlideMoving) && (n.leaveTimer = window.setTimeout(n.handleDragEnd, n.props.leaveTimeOut));
    }), Je(n, "handleEnterSlideOrTraveller", function() {
      n.setState({
        isTextActive: !0
      });
    }), Je(n, "handleLeaveSlideOrTraveller", function() {
      n.setState({
        isTextActive: !1
      });
    }), Je(n, "handleSlideDragStart", function(i) {
      var a = fb(i) ? i.changedTouches[0] : i;
      n.setState({
        isTravellerMoving: !1,
        isSlideMoving: !0,
        slideMoveStartX: a.pageX
      }), n.attachDragEndListener();
    }), n.travellerDragStartHandlers = {
      startX: n.handleTravellerDragStart.bind(n, "startX"),
      endX: n.handleTravellerDragStart.bind(n, "endX")
    }, n.state = {}, n;
  }
  return JR(t, e), VR(t, [{
    key: "componentWillUnmount",
    value: function() {
      this.leaveTimer && (clearTimeout(this.leaveTimer), this.leaveTimer = null), this.detachDragEndListener();
    }
  }, {
    key: "getIndex",
    value: function(n) {
      var i = n.startX, a = n.endX, o = this.state.scaleValues, u = this.props, s = u.gap, c = u.data, f = c.length - 1, l = Math.min(i, a), d = Math.max(i, a), p = t.getIndexInRange(o, l), y = t.getIndexInRange(o, d);
      return {
        startIndex: p - p % s,
        endIndex: y === f ? f : y - y % s
      };
    }
  }, {
    key: "getTextOfTick",
    value: function(n) {
      var i = this.props, a = i.data, o = i.tickFormatter, u = i.dataKey, s = tt(a[n], u, n);
      return ue(o) ? o(s, n) : s;
    }
  }, {
    key: "attachDragEndListener",
    value: function() {
      window.addEventListener("mouseup", this.handleDragEnd, !0), window.addEventListener("touchend", this.handleDragEnd, !0), window.addEventListener("mousemove", this.handleDrag, !0);
    }
  }, {
    key: "detachDragEndListener",
    value: function() {
      window.removeEventListener("mouseup", this.handleDragEnd, !0), window.removeEventListener("touchend", this.handleDragEnd, !0), window.removeEventListener("mousemove", this.handleDrag, !0);
    }
  }, {
    key: "handleSlideDrag",
    value: function(n) {
      var i = this.state, a = i.slideMoveStartX, o = i.startX, u = i.endX, s = this.props, c = s.x, f = s.width, l = s.travellerWidth, d = s.startIndex, p = s.endIndex, y = s.onChange, v = n.pageX - a;
      v > 0 ? v = Math.min(v, c + f - l - u, c + f - l - o) : v < 0 && (v = Math.max(v, c - o, c - u));
      var h = this.getIndex({
        startX: o + v,
        endX: u + v
      });
      (h.startIndex !== d || h.endIndex !== p) && y && y(h), this.setState({
        startX: o + v,
        endX: u + v,
        slideMoveStartX: n.pageX
      });
    }
  }, {
    key: "handleTravellerDragStart",
    value: function(n, i) {
      var a = fb(i) ? i.changedTouches[0] : i;
      this.setState({
        isSlideMoving: !1,
        isTravellerMoving: !0,
        movingTravellerId: n,
        brushMoveStartX: a.pageX
      }), this.attachDragEndListener();
    }
  }, {
    key: "handleTravellerMove",
    value: function(n) {
      var i = this.state, a = i.brushMoveStartX, o = i.movingTravellerId, u = i.endX, s = i.startX, c = this.state[o], f = this.props, l = f.x, d = f.width, p = f.travellerWidth, y = f.onChange, v = f.gap, h = f.data, g = {
        startX: this.state.startX,
        endX: this.state.endX
      }, w = n.pageX - a;
      w > 0 ? w = Math.min(w, l + d - p - c) : w < 0 && (w = Math.max(w, l - c)), g[o] = c + w;
      var b = this.getIndex(g), O = b.startIndex, m = b.endIndex, x = function() {
        var P = h.length - 1;
        return o === "startX" && (u > s ? O % v === 0 : m % v === 0) || u < s && m === P || o === "endX" && (u > s ? m % v === 0 : O % v === 0) || u > s && m === P;
      };
      this.setState(Je(Je({}, o, c + w), "brushMoveStartX", n.pageX), function() {
        y && x() && y(b);
      });
    }
  }, {
    key: "handleTravellerMoveKeyboard",
    value: function(n, i) {
      var a = this, o = this.state, u = o.scaleValues, s = o.startX, c = o.endX, f = this.state[i], l = u.indexOf(f);
      if (l !== -1) {
        var d = l + n;
        if (!(d === -1 || d >= u.length)) {
          var p = u[d];
          i === "startX" && p >= c || i === "endX" && p <= s || this.setState(Je({}, i, p), function() {
            a.props.onChange(a.getIndex({
              startX: a.state.startX,
              endX: a.state.endX
            }));
          });
        }
      }
    }
  }, {
    key: "renderBackground",
    value: function() {
      var n = this.props, i = n.x, a = n.y, o = n.width, u = n.height, s = n.fill, c = n.stroke;
      return /* @__PURE__ */ T.createElement("rect", {
        stroke: c,
        fill: s,
        x: i,
        y: a,
        width: o,
        height: u
      });
    }
  }, {
    key: "renderPanorama",
    value: function() {
      var n = this.props, i = n.x, a = n.y, o = n.width, u = n.height, s = n.data, c = n.children, f = n.padding, l = fr.only(c);
      return l ? /* @__PURE__ */ T.cloneElement(l, {
        x: i,
        y: a,
        width: o,
        height: u,
        margin: f,
        compact: !0,
        data: s
      }) : null;
    }
  }, {
    key: "renderTravellerLayer",
    value: function(n, i) {
      var a, o, u = this, s = this.props, c = s.y, f = s.travellerWidth, l = s.height, d = s.traveller, p = s.ariaLabel, y = s.data, v = s.startIndex, h = s.endIndex, g = Math.max(n, this.props.x), w = cl(cl({}, fe(this.props, !1)), {}, {
        x: g,
        y: c,
        width: f,
        height: l
      }), b = p || "Min value: ".concat((a = y[v]) === null || a === void 0 ? void 0 : a.name, ", Max value: ").concat((o = y[h]) === null || o === void 0 ? void 0 : o.name);
      return /* @__PURE__ */ T.createElement(je, {
        tabIndex: 0,
        role: "slider",
        "aria-label": b,
        "aria-valuenow": n,
        className: "recharts-brush-traveller",
        onMouseEnter: this.handleEnterSlideOrTraveller,
        onMouseLeave: this.handleLeaveSlideOrTraveller,
        onMouseDown: this.travellerDragStartHandlers[i],
        onTouchStart: this.travellerDragStartHandlers[i],
        onKeyDown: function(m) {
          ["ArrowLeft", "ArrowRight"].includes(m.key) && (m.preventDefault(), m.stopPropagation(), u.handleTravellerMoveKeyboard(m.key === "ArrowRight" ? 1 : -1, i));
        },
        onFocus: function() {
          u.setState({
            isTravellerFocused: !0
          });
        },
        onBlur: function() {
          u.setState({
            isTravellerFocused: !1
          });
        },
        style: {
          cursor: "col-resize"
        }
      }, t.renderTraveller(d, w));
    }
  }, {
    key: "renderSlide",
    value: function(n, i) {
      var a = this.props, o = a.y, u = a.height, s = a.stroke, c = a.travellerWidth, f = Math.min(n, i) + c, l = Math.max(Math.abs(i - n) - c, 0);
      return /* @__PURE__ */ T.createElement("rect", {
        className: "recharts-brush-slide",
        onMouseEnter: this.handleEnterSlideOrTraveller,
        onMouseLeave: this.handleLeaveSlideOrTraveller,
        onMouseDown: this.handleSlideDragStart,
        onTouchStart: this.handleSlideDragStart,
        style: {
          cursor: "move"
        },
        stroke: "none",
        fill: s,
        fillOpacity: 0.2,
        x: f,
        y: o,
        width: l,
        height: u
      });
    }
  }, {
    key: "renderText",
    value: function() {
      var n = this.props, i = n.startIndex, a = n.endIndex, o = n.y, u = n.height, s = n.travellerWidth, c = n.stroke, f = this.state, l = f.startX, d = f.endX, p = 5, y = {
        pointerEvents: "none",
        fill: c
      };
      return /* @__PURE__ */ T.createElement(je, {
        className: "recharts-brush-texts"
      }, /* @__PURE__ */ T.createElement(na, Ta({
        textAnchor: "end",
        verticalAnchor: "middle",
        x: Math.min(l, d) - p,
        y: o + u / 2
      }, y), this.getTextOfTick(i)), /* @__PURE__ */ T.createElement(na, Ta({
        textAnchor: "start",
        verticalAnchor: "middle",
        x: Math.max(l, d) + s + p,
        y: o + u / 2
      }, y), this.getTextOfTick(a)));
    }
  }, {
    key: "render",
    value: function() {
      var n = this.props, i = n.data, a = n.className, o = n.children, u = n.x, s = n.y, c = n.width, f = n.height, l = n.alwaysShowText, d = this.state, p = d.startX, y = d.endX, v = d.isTextActive, h = d.isSlideMoving, g = d.isTravellerMoving, w = d.isTravellerFocused;
      if (!i || !i.length || !K(u) || !K(s) || !K(c) || !K(f) || c <= 0 || f <= 0)
        return null;
      var b = pe("recharts-brush", a), O = T.Children.count(o) === 1, m = HR("userSelect", "none");
      return /* @__PURE__ */ T.createElement(je, {
        className: b,
        onMouseLeave: this.handleLeaveWrapper,
        onTouchMove: this.handleTouchMove,
        style: m
      }, this.renderBackground(), O && this.renderPanorama(), this.renderSlide(p, y), this.renderTravellerLayer(p, "startX"), this.renderTravellerLayer(y, "endX"), (v || h || g || w || l) && this.renderText());
    }
  }], [{
    key: "renderDefaultTraveller",
    value: function(n) {
      var i = n.x, a = n.y, o = n.width, u = n.height, s = n.stroke, c = Math.floor(a + u / 2) - 1;
      return /* @__PURE__ */ T.createElement(T.Fragment, null, /* @__PURE__ */ T.createElement("rect", {
        x: i,
        y: a,
        width: o,
        height: u,
        fill: s,
        stroke: "none"
      }), /* @__PURE__ */ T.createElement("line", {
        x1: i + 1,
        y1: c,
        x2: i + o - 1,
        y2: c,
        fill: "none",
        stroke: "#fff"
      }), /* @__PURE__ */ T.createElement("line", {
        x1: i + 1,
        y1: c + 2,
        x2: i + o - 1,
        y2: c + 2,
        fill: "none",
        stroke: "#fff"
      }));
    }
  }, {
    key: "renderTraveller",
    value: function(n, i) {
      var a;
      return /* @__PURE__ */ T.isValidElement(n) ? a = /* @__PURE__ */ T.cloneElement(n, i) : ue(n) ? a = n(i) : a = t.renderDefaultTraveller(i), a;
    }
  }, {
    key: "getDerivedStateFromProps",
    value: function(n, i) {
      var a = n.data, o = n.width, u = n.x, s = n.travellerWidth, c = n.updateId, f = n.startIndex, l = n.endIndex;
      if (a !== i.prevData || c !== i.prevUpdateId)
        return cl({
          prevData: a,
          prevTravellerWidth: s,
          prevUpdateId: c,
          prevX: u,
          prevWidth: o
        }, a && a.length ? ek({
          data: a,
          width: o,
          x: u,
          travellerWidth: s,
          startIndex: f,
          endIndex: l
        }) : {
          scale: null,
          scaleValues: null
        });
      if (i.scale && (o !== i.prevWidth || u !== i.prevX || s !== i.prevTravellerWidth)) {
        i.scale.range([u, u + o - s]);
        var d = i.scale.domain().map(function(p) {
          return i.scale(p);
        });
        return {
          prevData: a,
          prevTravellerWidth: s,
          prevUpdateId: c,
          prevX: u,
          prevWidth: o,
          startX: i.scale(n.startIndex),
          endX: i.scale(n.endIndex),
          scaleValues: d
        };
      }
      return null;
    }
  }, {
    key: "getIndexInRange",
    value: function(n, i) {
      for (var a = n.length, o = 0, u = a - 1; u - o > 1; ) {
        var s = Math.floor((o + u) / 2);
        n[s] > i ? u = s : o = s;
      }
      return i >= n[u] ? u : o;
    }
  }]);
}(Xt);
Je(Kr, "displayName", "Brush");
Je(Kr, "defaultProps", {
  height: 40,
  travellerWidth: 5,
  gap: 1,
  fill: "#fff",
  stroke: "#666",
  padding: {
    top: 1,
    right: 1,
    bottom: 1,
    left: 1
  },
  leaveTimeOut: 1e3,
  alwaysShowText: !1
});
var ll, db;
function tk() {
  if (db) return ll;
  db = 1;
  var e = yd();
  function t(r, n) {
    var i;
    return e(r, function(a, o, u) {
      return i = n(a, o, u), !i;
    }), !!i;
  }
  return ll = t, ll;
}
var fl, pb;
function rk() {
  if (pb) return fl;
  pb = 1;
  var e = q0(), t = Jt(), r = tk(), n = Xe(), i = eo();
  function a(o, u, s) {
    var c = n(o) ? e : r;
    return s && i(o, u, s) && (u = void 0), c(o, t(u, 3));
  }
  return fl = a, fl;
}
var nk = rk();
const ik = /* @__PURE__ */ Pe(nk);
var Ot = function(t, r) {
  var n = t.alwaysShow, i = t.ifOverflow;
  return n && (i = "extendDomain"), i === r;
}, dl, hb;
function ak() {
  if (hb) return dl;
  hb = 1;
  var e = rx();
  function t(r, n, i) {
    n == "__proto__" && e ? e(r, n, {
      configurable: !0,
      enumerable: !0,
      value: i,
      writable: !0
    }) : r[n] = i;
  }
  return dl = t, dl;
}
var pl, vb;
function ok() {
  if (vb) return pl;
  vb = 1;
  var e = ak(), t = ex(), r = Jt();
  function n(i, a) {
    var o = {};
    return a = r(a, 3), t(i, function(u, s, c) {
      e(o, s, a(u, s, c));
    }), o;
  }
  return pl = n, pl;
}
var uk = ok();
const sk = /* @__PURE__ */ Pe(uk);
var hl, yb;
function ck() {
  if (yb) return hl;
  yb = 1;
  function e(t, r) {
    for (var n = -1, i = t == null ? 0 : t.length; ++n < i; )
      if (!r(t[n], n, t))
        return !1;
    return !0;
  }
  return hl = e, hl;
}
var vl, mb;
function lk() {
  if (mb) return vl;
  mb = 1;
  var e = yd();
  function t(r, n) {
    var i = !0;
    return e(r, function(a, o, u) {
      return i = !!n(a, o, u), i;
    }), i;
  }
  return vl = t, vl;
}
var yl, gb;
function fk() {
  if (gb) return yl;
  gb = 1;
  var e = ck(), t = lk(), r = Jt(), n = Xe(), i = eo();
  function a(o, u, s) {
    var c = n(o) ? e : t;
    return s && i(o, u, s) && (u = void 0), c(o, r(u, 3));
  }
  return yl = a, yl;
}
var dk = fk();
const Mw = /* @__PURE__ */ Pe(dk);
var pk = ["x", "y"];
function Vr(e) {
  "@babel/helpers - typeof";
  return Vr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Vr(e);
}
function Af() {
  return Af = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Af.apply(this, arguments);
}
function bb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function En(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? bb(Object(r), !0).forEach(function(n) {
      hk(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : bb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function hk(e, t, r) {
  return t = vk(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function vk(e) {
  var t = yk(e, "string");
  return Vr(t) == "symbol" ? t : t + "";
}
function yk(e, t) {
  if (Vr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Vr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function mk(e, t) {
  if (e == null) return {};
  var r = gk(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function gk(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function bk(e, t) {
  var r = e.x, n = e.y, i = mk(e, pk), a = "".concat(r), o = parseInt(a, 10), u = "".concat(n), s = parseInt(u, 10), c = "".concat(t.height || i.height), f = parseInt(c, 10), l = "".concat(t.width || i.width), d = parseInt(l, 10);
  return En(En(En(En(En({}, t), i), o ? {
    x: o
  } : {}), s ? {
    y: s
  } : {}), {}, {
    height: f,
    width: d,
    name: t.name,
    radius: t.radius
  });
}
function xb(e) {
  return /* @__PURE__ */ T.createElement(MR, Af({
    shapeType: "rectangle",
    propTransformer: bk,
    activeClassName: "recharts-active-bar"
  }, e));
}
var xk = function(t) {
  var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0;
  return function(n, i) {
    if (typeof t == "number") return t;
    var a = typeof n == "number";
    return a ? t(n, i) : (a || (process.env.NODE_ENV !== "production" ? Ye(!1, "minPointSize callback function received a value with type of ".concat(Vr(n), ". Currently only numbers are supported.")) : Ye()), r);
  };
}, wk = ["value", "background"], Iw;
function Yr(e) {
  "@babel/helpers - typeof";
  return Yr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Yr(e);
}
function Ok(e, t) {
  if (e == null) return {};
  var r = _k(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function _k(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function Ca() {
  return Ca = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Ca.apply(this, arguments);
}
function wb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Ie(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? wb(Object(r), !0).forEach(function(n) {
      Gt(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : wb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Sk(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function Ob(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Rw(n.key), n);
  }
}
function Pk(e, t, r) {
  return Ob(e.prototype, t), Ob(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function Ak(e, t, r) {
  return t = Ma(t), Ek(e, $w() ? Reflect.construct(t, r, Ma(e).constructor) : t.apply(e, r));
}
function Ek(e, t) {
  if (t && (Yr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return Tk(e);
}
function Tk(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function $w() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return ($w = function() {
    return !!e;
  })();
}
function Ma(e) {
  return Ma = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Ma(e);
}
function jk(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Ef(e, t);
}
function Ef(e, t) {
  return Ef = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Ef(e, t);
}
function Gt(e, t, r) {
  return t = Rw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Rw(e) {
  var t = Ck(e, "string");
  return Yr(t) == "symbol" ? t : t + "";
}
function Ck(e, t) {
  if (Yr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Yr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Pi = /* @__PURE__ */ function(e) {
  function t() {
    var r;
    Sk(this, t);
    for (var n = arguments.length, i = new Array(n), a = 0; a < n; a++)
      i[a] = arguments[a];
    return r = Ak(this, t, [].concat(i)), Gt(r, "state", {
      isAnimationFinished: !1
    }), Gt(r, "id", gi("recharts-bar-")), Gt(r, "handleAnimationEnd", function() {
      var o = r.props.onAnimationEnd;
      r.setState({
        isAnimationFinished: !0
      }), o && o();
    }), Gt(r, "handleAnimationStart", function() {
      var o = r.props.onAnimationStart;
      r.setState({
        isAnimationFinished: !1
      }), o && o();
    }), r;
  }
  return jk(t, e), Pk(t, [{
    key: "renderRectanglesStatically",
    value: function(n) {
      var i = this, a = this.props, o = a.shape, u = a.dataKey, s = a.activeIndex, c = a.activeBar, f = fe(this.props, !1);
      return n && n.map(function(l, d) {
        var p = d === s, y = p ? c : o, v = Ie(Ie(Ie({}, f), l), {}, {
          isActive: p,
          option: y,
          index: d,
          dataKey: u,
          onAnimationStart: i.handleAnimationStart,
          onAnimationEnd: i.handleAnimationEnd
        });
        return /* @__PURE__ */ T.createElement(je, Ca({
          className: "recharts-bar-rectangle"
        }, Hi(i.props, l, d), {
          key: "rectangle-".concat(l == null ? void 0 : l.x, "-").concat(l == null ? void 0 : l.y, "-").concat(l == null ? void 0 : l.value)
        }), /* @__PURE__ */ T.createElement(xb, v));
      });
    }
  }, {
    key: "renderRectanglesWithAnimation",
    value: function() {
      var n = this, i = this.props, a = i.data, o = i.layout, u = i.isAnimationActive, s = i.animationBegin, c = i.animationDuration, f = i.animationEasing, l = i.animationId, d = this.state.prevData;
      return /* @__PURE__ */ T.createElement(Dt, {
        begin: s,
        duration: c,
        isActive: u,
        easing: f,
        from: {
          t: 0
        },
        to: {
          t: 1
        },
        key: "bar-".concat(l),
        onAnimationEnd: this.handleAnimationEnd,
        onAnimationStart: this.handleAnimationStart
      }, function(p) {
        var y = p.t, v = a.map(function(h, g) {
          var w = d && d[g];
          if (w) {
            var b = ht(w.x, h.x), O = ht(w.y, h.y), m = ht(w.width, h.width), x = ht(w.height, h.height);
            return Ie(Ie({}, h), {}, {
              x: b(y),
              y: O(y),
              width: m(y),
              height: x(y)
            });
          }
          if (o === "horizontal") {
            var _ = ht(0, h.height), P = _(y);
            return Ie(Ie({}, h), {}, {
              y: h.y + h.height - P,
              height: P
            });
          }
          var E = ht(0, h.width), I = E(y);
          return Ie(Ie({}, h), {}, {
            width: I
          });
        });
        return /* @__PURE__ */ T.createElement(je, null, n.renderRectanglesStatically(v));
      });
    }
  }, {
    key: "renderRectangles",
    value: function() {
      var n = this.props, i = n.data, a = n.isAnimationActive, o = this.state.prevData;
      return a && i && i.length && (!o || !co(o, i)) ? this.renderRectanglesWithAnimation() : this.renderRectanglesStatically(i);
    }
  }, {
    key: "renderBackground",
    value: function() {
      var n = this, i = this.props, a = i.data, o = i.dataKey, u = i.activeIndex, s = fe(this.props.background, !1);
      return a.map(function(c, f) {
        c.value;
        var l = c.background, d = Ok(c, wk);
        if (!l)
          return null;
        var p = Ie(Ie(Ie(Ie(Ie({}, d), {}, {
          fill: "#eee"
        }, l), s), Hi(n.props, c, f)), {}, {
          onAnimationStart: n.handleAnimationStart,
          onAnimationEnd: n.handleAnimationEnd,
          dataKey: o,
          index: f,
          className: "recharts-bar-background-rectangle"
        });
        return /* @__PURE__ */ T.createElement(xb, Ca({
          key: "background-bar-".concat(f),
          option: n.props.background,
          isActive: f === u
        }, p));
      });
    }
  }, {
    key: "renderErrorBar",
    value: function(n, i) {
      if (this.props.isAnimationActive && !this.state.isAnimationFinished)
        return null;
      var a = this.props, o = a.data, u = a.xAxis, s = a.yAxis, c = a.layout, f = a.children, l = ot(f, Si);
      if (!l)
        return null;
      var d = c === "vertical" ? o[0].height / 2 : o[0].width / 2, p = function(h, g) {
        var w = Array.isArray(h.value) ? h.value[1] : h.value;
        return {
          x: h.x,
          y: h.y,
          value: w,
          errorVal: tt(h, g)
        };
      }, y = {
        clipPath: n ? "url(#clipPath-".concat(i, ")") : null
      };
      return /* @__PURE__ */ T.createElement(je, y, l.map(function(v) {
        return /* @__PURE__ */ T.cloneElement(v, {
          key: "error-bar-".concat(i, "-").concat(v.props.dataKey),
          data: o,
          xAxis: u,
          yAxis: s,
          layout: c,
          offset: d,
          dataPointFormatter: p
        });
      }));
    }
  }, {
    key: "render",
    value: function() {
      var n = this.props, i = n.hide, a = n.data, o = n.className, u = n.xAxis, s = n.yAxis, c = n.left, f = n.top, l = n.width, d = n.height, p = n.isAnimationActive, y = n.background, v = n.id;
      if (i || !a || !a.length)
        return null;
      var h = this.state.isAnimationFinished, g = pe("recharts-bar", o), w = u && u.allowDataOverflow, b = s && s.allowDataOverflow, O = w || b, m = ce(v) ? this.id : v;
      return /* @__PURE__ */ T.createElement(je, {
        className: g
      }, w || b ? /* @__PURE__ */ T.createElement("defs", null, /* @__PURE__ */ T.createElement("clipPath", {
        id: "clipPath-".concat(m)
      }, /* @__PURE__ */ T.createElement("rect", {
        x: w ? c : c - l / 2,
        y: b ? f : f - d / 2,
        width: w ? l : l * 2,
        height: b ? d : d * 2
      }))) : null, /* @__PURE__ */ T.createElement(je, {
        className: "recharts-bar-rectangles",
        clipPath: O ? "url(#clipPath-".concat(m, ")") : null
      }, y ? this.renderBackground() : null, this.renderRectangles()), this.renderErrorBar(O, m), (!p || h) && Vt.renderCallByParent(this.props, a));
    }
  }], [{
    key: "getDerivedStateFromProps",
    value: function(n, i) {
      return n.animationId !== i.prevAnimationId ? {
        prevAnimationId: n.animationId,
        curData: n.data,
        prevData: i.curData
      } : n.data !== i.curData ? {
        curData: n.data
      } : null;
    }
  }]);
}(Xt);
Iw = Pi;
Gt(Pi, "displayName", "Bar");
Gt(Pi, "defaultProps", {
  xAxisId: 0,
  yAxisId: 0,
  legendType: "rect",
  minPointSize: 0,
  hide: !1,
  data: [],
  layout: "vertical",
  activeBar: !1,
  isAnimationActive: !It.isSsr,
  animationBegin: 0,
  animationDuration: 400,
  animationEasing: "ease"
});
Gt(Pi, "getComposedData", function(e) {
  var t = e.props, r = e.item, n = e.barPosition, i = e.bandSize, a = e.xAxis, o = e.yAxis, u = e.xAxisTicks, s = e.yAxisTicks, c = e.stackedData, f = e.dataStartIndex, l = e.displayedData, d = e.offset, p = WC(n, r);
  if (!p)
    return null;
  var y = t.layout, v = r.type.defaultProps, h = v !== void 0 ? Ie(Ie({}, v), r.props) : r.props, g = h.dataKey, w = h.children, b = h.minPointSize, O = y === "horizontal" ? o : a, m = c ? O.scale.domain() : null, x = JC({
    numericAxis: O
  }), _ = ot(w, sx), P = l.map(function(E, I) {
    var S, j, M, R, k, q;
    c ? S = GC(c[f + I], m) : (S = tt(E, g), Array.isArray(S) || (S = [x, S]));
    var L = xk(b, Iw.defaultProps.minPointSize)(S[1], I);
    if (y === "horizontal") {
      var U, z = [o.scale(S[0]), o.scale(S[1])], $ = z[0], D = z[1];
      j = Jm({
        axis: a,
        ticks: u,
        bandSize: i,
        offset: p.offset,
        entry: E,
        index: I
      }), M = (U = D ?? $) !== null && U !== void 0 ? U : void 0, R = p.size;
      var B = $ - D;
      if (k = Number.isNaN(B) ? 0 : B, q = {
        x: j,
        y: o.y,
        width: R,
        height: o.height
      }, Math.abs(L) > 0 && Math.abs(k) < Math.abs(L)) {
        var G = yt(k || L) * (Math.abs(L) - Math.abs(k));
        M -= G, k += G;
      }
    } else {
      var V = [a.scale(S[0]), a.scale(S[1])], te = V[0], re = V[1];
      if (j = te, M = Jm({
        axis: o,
        ticks: s,
        bandSize: i,
        offset: p.offset,
        entry: E,
        index: I
      }), R = re - te, k = p.size, q = {
        x: a.x,
        y: M,
        width: a.width,
        height: k
      }, Math.abs(L) > 0 && Math.abs(R) < Math.abs(L)) {
        var ae = yt(R || L) * (Math.abs(L) - Math.abs(R));
        R += ae;
      }
    }
    return Ie(Ie(Ie({}, E), {}, {
      x: j,
      y: M,
      width: R,
      height: k,
      value: c ? S : S[1],
      payload: E,
      background: q
    }, _[I] && _[I].props), {}, {
      tooltipPayload: [hw(r, E)],
      tooltipPosition: {
        x: j + R / 2,
        y: M + k / 2
      }
    });
  });
  return Ie({
    data: P,
    layout: y
  }, d);
});
function fi(e) {
  "@babel/helpers - typeof";
  return fi = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, fi(e);
}
function Mk(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function _b(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, kw(n.key), n);
  }
}
function Ik(e, t, r) {
  return _b(e.prototype, t), _b(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function Sb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function dt(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Sb(Object(r), !0).forEach(function(n) {
      vo(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Sb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function vo(e, t, r) {
  return t = kw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function kw(e) {
  var t = $k(e, "string");
  return fi(t) == "symbol" ? t : t + "";
}
function $k(e, t) {
  if (fi(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (fi(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Rk = function(t, r, n, i, a) {
  var o = t.width, u = t.height, s = t.layout, c = t.children, f = Object.keys(r), l = {
    left: n.left,
    leftMirror: n.left,
    right: o - n.right,
    rightMirror: o - n.right,
    top: n.top,
    topMirror: n.top,
    bottom: u - n.bottom,
    bottomMirror: u - n.bottom
  }, d = !!Qe(c, Pi);
  return f.reduce(function(p, y) {
    var v = r[y], h = v.orientation, g = v.domain, w = v.padding, b = w === void 0 ? {} : w, O = v.mirror, m = v.reversed, x = "".concat(h).concat(O ? "Mirror" : ""), _, P, E, I, S;
    if (v.type === "number" && (v.padding === "gap" || v.padding === "no-gap")) {
      var j = g[1] - g[0], M = 1 / 0, R = v.categoricalDomain.sort();
      if (R.forEach(function(V, te) {
        te > 0 && (M = Math.min((V || 0) - (R[te - 1] || 0), M));
      }), Number.isFinite(M)) {
        var k = M / j, q = v.layout === "vertical" ? n.height : n.width;
        if (v.padding === "gap" && (_ = k * q / 2), v.padding === "no-gap") {
          var L = hr(t.barCategoryGap, k * q), U = k * q / 2;
          _ = U - L - (U - L) / q * L;
        }
      }
    }
    i === "xAxis" ? P = [n.left + (b.left || 0) + (_ || 0), n.left + n.width - (b.right || 0) - (_ || 0)] : i === "yAxis" ? P = s === "horizontal" ? [n.top + n.height - (b.bottom || 0), n.top + (b.top || 0)] : [n.top + (b.top || 0) + (_ || 0), n.top + n.height - (b.bottom || 0) - (_ || 0)] : P = v.range, m && (P = [P[1], P[0]]);
    var z = zC(v, a, d), $ = z.scale, D = z.realScaleType;
    $.domain(g).range(P), UC($);
    var B = ZC($, dt(dt({}, v), {}, {
      realScaleType: D
    }));
    i === "xAxis" ? (S = h === "top" && !O || h === "bottom" && O, E = n.left, I = l[x] - S * v.height) : i === "yAxis" && (S = h === "left" && !O || h === "right" && O, E = l[x] - S * v.width, I = n.top);
    var G = dt(dt(dt({}, v), B), {}, {
      realScaleType: D,
      x: E,
      y: I,
      scale: $,
      width: i === "xAxis" ? n.width : v.width,
      height: i === "yAxis" ? n.height : v.height
    });
    return G.bandSize = ma(G, B), !v.hide && i === "xAxis" ? l[x] += (S ? -1 : 1) * G.height : v.hide || (l[x] += (S ? -1 : 1) * G.width), dt(dt({}, p), {}, vo({}, y, G));
  }, {});
}, Nw = function(t, r) {
  var n = t.x, i = t.y, a = r.x, o = r.y;
  return {
    x: Math.min(n, a),
    y: Math.min(i, o),
    width: Math.abs(a - n),
    height: Math.abs(o - i)
  };
}, kk = function(t) {
  var r = t.x1, n = t.y1, i = t.x2, a = t.y2;
  return Nw({
    x: r,
    y: n
  }, {
    x: i,
    y: a
  });
}, Dw = /* @__PURE__ */ function() {
  function e(t) {
    Mk(this, e), this.scale = t;
  }
  return Ik(e, [{
    key: "domain",
    get: function() {
      return this.scale.domain;
    }
  }, {
    key: "range",
    get: function() {
      return this.scale.range;
    }
  }, {
    key: "rangeMin",
    get: function() {
      return this.range()[0];
    }
  }, {
    key: "rangeMax",
    get: function() {
      return this.range()[1];
    }
  }, {
    key: "bandwidth",
    get: function() {
      return this.scale.bandwidth;
    }
  }, {
    key: "apply",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, i = n.bandAware, a = n.position;
      if (r !== void 0) {
        if (a)
          switch (a) {
            case "start":
              return this.scale(r);
            case "middle": {
              var o = this.bandwidth ? this.bandwidth() / 2 : 0;
              return this.scale(r) + o;
            }
            case "end": {
              var u = this.bandwidth ? this.bandwidth() : 0;
              return this.scale(r) + u;
            }
            default:
              return this.scale(r);
          }
        if (i) {
          var s = this.bandwidth ? this.bandwidth() / 2 : 0;
          return this.scale(r) + s;
        }
        return this.scale(r);
      }
    }
  }, {
    key: "isInRange",
    value: function(r) {
      var n = this.range(), i = n[0], a = n[n.length - 1];
      return i <= a ? r >= i && r <= a : r >= a && r <= i;
    }
  }], [{
    key: "create",
    value: function(r) {
      return new e(r);
    }
  }]);
}();
vo(Dw, "EPS", 1e-4);
var Kd = function(t) {
  var r = Object.keys(t).reduce(function(n, i) {
    return dt(dt({}, n), {}, vo({}, i, Dw.create(t[i])));
  }, {});
  return dt(dt({}, r), {}, {
    apply: function(i) {
      var a = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, o = a.bandAware, u = a.position;
      return sk(i, function(s, c) {
        return r[c].apply(s, {
          bandAware: o,
          position: u
        });
      });
    },
    isInRange: function(i) {
      return Mw(i, function(a, o) {
        return r[o].isInRange(a);
      });
    }
  });
};
function Nk(e) {
  return (e % 180 + 180) % 180;
}
var Dk = function(t) {
  var r = t.width, n = t.height, i = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 0, a = Nk(i), o = a * Math.PI / 180, u = Math.atan(n / r), s = o > u && o < Math.PI - u ? n / Math.sin(o) : r / Math.cos(o);
  return Math.abs(s);
}, ml, Pb;
function qk() {
  if (Pb) return ml;
  Pb = 1;
  var e = Jt(), t = bi(), r = Ja();
  function n(i) {
    return function(a, o, u) {
      var s = Object(a);
      if (!t(a)) {
        var c = e(o, 3);
        a = r(a), o = function(l) {
          return c(s[l], l, s);
        };
      }
      var f = i(a, o, u);
      return f > -1 ? s[c ? a[f] : f] : void 0;
    };
  }
  return ml = n, ml;
}
var gl, Ab;
function Lk() {
  if (Ab) return gl;
  Ab = 1;
  var e = Ew();
  function t(r) {
    var n = e(r), i = n % 1;
    return n === n ? i ? n - i : n : 0;
  }
  return gl = t, gl;
}
var bl, Eb;
function Bk() {
  if (Eb) return bl;
  Eb = 1;
  var e = Y0(), t = Jt(), r = Lk(), n = Math.max;
  function i(a, o, u) {
    var s = a == null ? 0 : a.length;
    if (!s)
      return -1;
    var c = u == null ? 0 : r(u);
    return c < 0 && (c = n(s + c, 0)), e(a, t(o, 3), c);
  }
  return bl = i, bl;
}
var xl, Tb;
function Fk() {
  if (Tb) return xl;
  Tb = 1;
  var e = qk(), t = Bk(), r = e(t);
  return xl = r, xl;
}
var zk = Fk();
const Uk = /* @__PURE__ */ Pe(zk);
var Wk = f0();
const Gk = /* @__PURE__ */ Pe(Wk);
var Hk = Gk(function(e) {
  return {
    x: e.left,
    y: e.top,
    width: e.width,
    height: e.height
  };
}, function(e) {
  return ["l", e.left, "t", e.top, "w", e.width, "h", e.height].join("");
});
function Ia(e) {
  "@babel/helpers - typeof";
  return Ia = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Ia(e);
}
var Vd = /* @__PURE__ */ mr(void 0), Yd = /* @__PURE__ */ mr(void 0), qw = /* @__PURE__ */ mr(void 0), Lw = /* @__PURE__ */ mr({}), Bw = /* @__PURE__ */ mr(void 0), Fw = /* @__PURE__ */ mr(0), zw = /* @__PURE__ */ mr(0), jb = function(t) {
  var r = t.state, n = r.xAxisMap, i = r.yAxisMap, a = r.offset, o = t.clipPathId, u = t.children, s = t.width, c = t.height, f = Hk(a);
  return /* @__PURE__ */ T.createElement(Vd.Provider, {
    value: n
  }, /* @__PURE__ */ T.createElement(Yd.Provider, {
    value: i
  }, /* @__PURE__ */ T.createElement(Lw.Provider, {
    value: a
  }, /* @__PURE__ */ T.createElement(qw.Provider, {
    value: f
  }, /* @__PURE__ */ T.createElement(Bw.Provider, {
    value: o
  }, /* @__PURE__ */ T.createElement(Fw.Provider, {
    value: c
  }, /* @__PURE__ */ T.createElement(zw.Provider, {
    value: s
  }, u)))))));
}, Kk = function() {
  return qt(Bw);
};
function Uw(e) {
  var t = Object.keys(e);
  return t.length === 0 ? "There are no available ids." : "Available ids are: ".concat(t, ".");
}
var Ww = function(t) {
  var r = qt(Vd);
  r == null && (process.env.NODE_ENV !== "production" ? Ye(!1, "Could not find Recharts context; are you sure this is rendered inside a Recharts wrapper component?") : Ye());
  var n = r[t];
  return n == null && (process.env.NODE_ENV !== "production" ? Ye(!1, 'Could not find xAxis by id "'.concat(t, '" [').concat(Ia(t), "]. ").concat(Uw(r))) : Ye()), n;
}, Vk = function() {
  var t = qt(Vd);
  return Wt(t);
}, Yk = function() {
  var t = qt(Yd), r = Uk(t, function(n) {
    return Mw(n.domain, Number.isFinite);
  });
  return r || Wt(t);
}, Gw = function(t) {
  var r = qt(Yd);
  r == null && (process.env.NODE_ENV !== "production" ? Ye(!1, "Could not find Recharts context; are you sure this is rendered inside a Recharts wrapper component?") : Ye());
  var n = r[t];
  return n == null && (process.env.NODE_ENV !== "production" ? Ye(!1, 'Could not find yAxis by id "'.concat(t, '" [').concat(Ia(t), "]. ").concat(Uw(r))) : Ye()), n;
}, Xk = function() {
  var t = qt(qw);
  return t;
}, Zk = function() {
  return qt(Lw);
}, Xd = function() {
  return qt(zw);
}, Zd = function() {
  return qt(Fw);
};
function Xr(e) {
  "@babel/helpers - typeof";
  return Xr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Xr(e);
}
function Jk(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function Qk(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Kw(n.key), n);
  }
}
function eN(e, t, r) {
  return Qk(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function tN(e, t, r) {
  return t = $a(t), rN(e, Hw() ? Reflect.construct(t, r || [], $a(e).constructor) : t.apply(e, r));
}
function rN(e, t) {
  if (t && (Xr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return nN(e);
}
function nN(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function Hw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (Hw = function() {
    return !!e;
  })();
}
function $a(e) {
  return $a = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, $a(e);
}
function iN(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Tf(e, t);
}
function Tf(e, t) {
  return Tf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Tf(e, t);
}
function Cb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Mb(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Cb(Object(r), !0).forEach(function(n) {
      Jd(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Cb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Jd(e, t, r) {
  return t = Kw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Kw(e) {
  var t = aN(e, "string");
  return Xr(t) == "symbol" ? t : t + "";
}
function aN(e, t) {
  if (Xr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Xr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function oN(e, t) {
  return lN(e) || cN(e, t) || sN(e, t) || uN();
}
function uN() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function sN(e, t) {
  if (e) {
    if (typeof e == "string") return Ib(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Ib(e, t);
  }
}
function Ib(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function cN(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function lN(e) {
  if (Array.isArray(e)) return e;
}
function jf() {
  return jf = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, jf.apply(this, arguments);
}
var fN = function(t, r) {
  var n;
  return /* @__PURE__ */ T.isValidElement(t) ? n = /* @__PURE__ */ T.cloneElement(t, r) : ue(t) ? n = t(r) : n = /* @__PURE__ */ T.createElement("line", jf({}, r, {
    className: "recharts-reference-line-line"
  })), n;
}, dN = function(t, r, n, i, a, o, u, s, c) {
  var f = a.x, l = a.y, d = a.width, p = a.height;
  if (n) {
    var y = c.y, v = t.y.apply(y, {
      position: o
    });
    if (Ot(c, "discard") && !t.y.isInRange(v))
      return null;
    var h = [{
      x: f + d,
      y: v
    }, {
      x: f,
      y: v
    }];
    return s === "left" ? h.reverse() : h;
  }
  if (r) {
    var g = c.x, w = t.x.apply(g, {
      position: o
    });
    if (Ot(c, "discard") && !t.x.isInRange(w))
      return null;
    var b = [{
      x: w,
      y: l + p
    }, {
      x: w,
      y: l
    }];
    return u === "top" ? b.reverse() : b;
  }
  if (i) {
    var O = c.segment, m = O.map(function(x) {
      return t.apply(x, {
        position: o
      });
    });
    return Ot(c, "discard") && ik(m, function(x) {
      return !t.isInRange(x);
    }) ? null : m;
  }
  return null;
};
function pN(e) {
  var t = e.x, r = e.y, n = e.segment, i = e.xAxisId, a = e.yAxisId, o = e.shape, u = e.className, s = e.alwaysShow, c = Kk(), f = Ww(i), l = Gw(a), d = Xk();
  if (!c || !d)
    return null;
  kr(s === void 0, 'The alwaysShow prop is deprecated. Please use ifOverflow="extendDomain" instead.');
  var p = Kd({
    x: f.scale,
    y: l.scale
  }), y = ke(t), v = ke(r), h = n && n.length === 2, g = dN(p, y, v, h, d, e.position, f.orientation, l.orientation, e);
  if (!g)
    return null;
  var w = oN(g, 2), b = w[0], O = b.x, m = b.y, x = w[1], _ = x.x, P = x.y, E = Ot(e, "hidden") ? "url(#".concat(c, ")") : void 0, I = Mb(Mb({
    clipPath: E
  }, fe(e, !0)), {}, {
    x1: O,
    y1: m,
    x2: _,
    y2: P
  });
  return /* @__PURE__ */ T.createElement(je, {
    className: pe("recharts-reference-line", u)
  }, fN(o, I), Ue.renderCallByParent(e, kk({
    x1: O,
    y1: m,
    x2: _,
    y2: P
  })));
}
var Qd = /* @__PURE__ */ function(e) {
  function t() {
    return Jk(this, t), tN(this, t, arguments);
  }
  return iN(t, e), eN(t, [{
    key: "render",
    value: function() {
      return /* @__PURE__ */ T.createElement(pN, this.props);
    }
  }]);
}(T.Component);
Jd(Qd, "displayName", "ReferenceLine");
Jd(Qd, "defaultProps", {
  isFront: !1,
  ifOverflow: "discard",
  xAxisId: 0,
  yAxisId: 0,
  fill: "none",
  stroke: "#ccc",
  fillOpacity: 1,
  strokeWidth: 1,
  position: "middle"
});
function Cf() {
  return Cf = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Cf.apply(this, arguments);
}
function Zr(e) {
  "@babel/helpers - typeof";
  return Zr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Zr(e);
}
function $b(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Rb(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? $b(Object(r), !0).forEach(function(n) {
      yo(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : $b(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function hN(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function vN(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Yw(n.key), n);
  }
}
function yN(e, t, r) {
  return vN(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function mN(e, t, r) {
  return t = Ra(t), gN(e, Vw() ? Reflect.construct(t, r || [], Ra(e).constructor) : t.apply(e, r));
}
function gN(e, t) {
  if (t && (Zr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return bN(e);
}
function bN(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function Vw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (Vw = function() {
    return !!e;
  })();
}
function Ra(e) {
  return Ra = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Ra(e);
}
function xN(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Mf(e, t);
}
function Mf(e, t) {
  return Mf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Mf(e, t);
}
function yo(e, t, r) {
  return t = Yw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Yw(e) {
  var t = wN(e, "string");
  return Zr(t) == "symbol" ? t : t + "";
}
function wN(e, t) {
  if (Zr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Zr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var ON = function(t) {
  var r = t.x, n = t.y, i = t.xAxis, a = t.yAxis, o = Kd({
    x: i.scale,
    y: a.scale
  }), u = o.apply({
    x: r,
    y: n
  }, {
    bandAware: !0
  });
  return Ot(t, "discard") && !o.isInRange(u) ? null : u;
}, mo = /* @__PURE__ */ function(e) {
  function t() {
    return hN(this, t), mN(this, t, arguments);
  }
  return xN(t, e), yN(t, [{
    key: "render",
    value: function() {
      var n = this.props, i = n.x, a = n.y, o = n.r, u = n.alwaysShow, s = n.clipPathId, c = ke(i), f = ke(a);
      if (kr(u === void 0, 'The alwaysShow prop is deprecated. Please use ifOverflow="extendDomain" instead.'), !c || !f)
        return null;
      var l = ON(this.props);
      if (!l)
        return null;
      var d = l.x, p = l.y, y = this.props, v = y.shape, h = y.className, g = Ot(this.props, "hidden") ? "url(#".concat(s, ")") : void 0, w = Rb(Rb({
        clipPath: g
      }, fe(this.props, !0)), {}, {
        cx: d,
        cy: p
      });
      return /* @__PURE__ */ T.createElement(je, {
        className: pe("recharts-reference-dot", h)
      }, t.renderDot(v, w), Ue.renderCallByParent(this.props, {
        x: d - o,
        y: p - o,
        width: 2 * o,
        height: 2 * o
      }));
    }
  }]);
}(T.Component);
yo(mo, "displayName", "ReferenceDot");
yo(mo, "defaultProps", {
  isFront: !1,
  ifOverflow: "discard",
  xAxisId: 0,
  yAxisId: 0,
  r: 10,
  fill: "#fff",
  stroke: "#ccc",
  fillOpacity: 1,
  strokeWidth: 1
});
yo(mo, "renderDot", function(e, t) {
  var r;
  return /* @__PURE__ */ T.isValidElement(e) ? r = /* @__PURE__ */ T.cloneElement(e, t) : ue(e) ? r = e(t) : r = /* @__PURE__ */ T.createElement(Hd, Cf({}, t, {
    cx: t.cx,
    cy: t.cy,
    className: "recharts-reference-dot-dot"
  })), r;
});
function If() {
  return If = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, If.apply(this, arguments);
}
function Jr(e) {
  "@babel/helpers - typeof";
  return Jr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Jr(e);
}
function kb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Nb(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? kb(Object(r), !0).forEach(function(n) {
      go(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : kb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function _N(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function SN(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, Zw(n.key), n);
  }
}
function PN(e, t, r) {
  return SN(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function AN(e, t, r) {
  return t = ka(t), EN(e, Xw() ? Reflect.construct(t, r || [], ka(e).constructor) : t.apply(e, r));
}
function EN(e, t) {
  if (t && (Jr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return TN(e);
}
function TN(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function Xw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (Xw = function() {
    return !!e;
  })();
}
function ka(e) {
  return ka = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, ka(e);
}
function jN(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && $f(e, t);
}
function $f(e, t) {
  return $f = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, $f(e, t);
}
function go(e, t, r) {
  return t = Zw(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function Zw(e) {
  var t = CN(e, "string");
  return Jr(t) == "symbol" ? t : t + "";
}
function CN(e, t) {
  if (Jr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Jr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var MN = function(t, r, n, i, a) {
  var o = a.x1, u = a.x2, s = a.y1, c = a.y2, f = a.xAxis, l = a.yAxis;
  if (!f || !l) return null;
  var d = Kd({
    x: f.scale,
    y: l.scale
  }), p = {
    x: t ? d.x.apply(o, {
      position: "start"
    }) : d.x.rangeMin,
    y: n ? d.y.apply(s, {
      position: "start"
    }) : d.y.rangeMin
  }, y = {
    x: r ? d.x.apply(u, {
      position: "end"
    }) : d.x.rangeMax,
    y: i ? d.y.apply(c, {
      position: "end"
    }) : d.y.rangeMax
  };
  return Ot(a, "discard") && (!d.isInRange(p) || !d.isInRange(y)) ? null : Nw(p, y);
}, bo = /* @__PURE__ */ function(e) {
  function t() {
    return _N(this, t), AN(this, t, arguments);
  }
  return jN(t, e), PN(t, [{
    key: "render",
    value: function() {
      var n = this.props, i = n.x1, a = n.x2, o = n.y1, u = n.y2, s = n.className, c = n.alwaysShow, f = n.clipPathId;
      kr(c === void 0, 'The alwaysShow prop is deprecated. Please use ifOverflow="extendDomain" instead.');
      var l = ke(i), d = ke(a), p = ke(o), y = ke(u), v = this.props.shape;
      if (!l && !d && !p && !y && !v)
        return null;
      var h = MN(l, d, p, y, this.props);
      if (!h && !v)
        return null;
      var g = Ot(this.props, "hidden") ? "url(#".concat(f, ")") : void 0;
      return /* @__PURE__ */ T.createElement(je, {
        className: pe("recharts-reference-area", s)
      }, t.renderRect(v, Nb(Nb({
        clipPath: g
      }, fe(this.props, !0)), h)), Ue.renderCallByParent(this.props, h));
    }
  }]);
}(T.Component);
go(bo, "displayName", "ReferenceArea");
go(bo, "defaultProps", {
  isFront: !1,
  ifOverflow: "discard",
  xAxisId: 0,
  yAxisId: 0,
  r: 10,
  fill: "#ccc",
  fillOpacity: 0.5,
  stroke: "none",
  strokeWidth: 1
});
go(bo, "renderRect", function(e, t) {
  var r;
  return /* @__PURE__ */ T.isValidElement(e) ? r = /* @__PURE__ */ T.cloneElement(e, t) : ue(e) ? r = e(t) : r = /* @__PURE__ */ T.createElement(Gd, If({}, t, {
    className: "recharts-reference-area-rect"
  })), r;
});
function Jw(e, t, r) {
  if (t < 1)
    return [];
  if (t === 1 && r === void 0)
    return e;
  for (var n = [], i = 0; i < e.length; i += t)
    n.push(e[i]);
  return n;
}
function IN(e, t, r) {
  var n = {
    width: e.width + t.width,
    height: e.height + t.height
  };
  return Dk(n, r);
}
function $N(e, t, r) {
  var n = r === "width", i = e.x, a = e.y, o = e.width, u = e.height;
  return t === 1 ? {
    start: n ? i : a,
    end: n ? i + o : a + u
  } : {
    start: n ? i + o : a + u,
    end: n ? i : a
  };
}
function Na(e, t, r, n, i) {
  if (e * t < e * n || e * t > e * i)
    return !1;
  var a = r();
  return e * (t - e * a / 2 - n) >= 0 && e * (t + e * a / 2 - i) <= 0;
}
function RN(e, t) {
  return Jw(e, t + 1);
}
function kN(e, t, r, n, i) {
  for (var a = (n || []).slice(), o = t.start, u = t.end, s = 0, c = 1, f = o, l = function() {
    var y = n == null ? void 0 : n[s];
    if (y === void 0)
      return {
        v: Jw(n, c)
      };
    var v = s, h, g = function() {
      return h === void 0 && (h = r(y, v)), h;
    }, w = y.coordinate, b = s === 0 || Na(e, w, g, f, u);
    b || (s = 0, f = o, c += 1), b && (f = w + e * (g() / 2 + i), s += c);
  }, d; c <= a.length; )
    if (d = l(), d) return d.v;
  return [];
}
function di(e) {
  "@babel/helpers - typeof";
  return di = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, di(e);
}
function Db(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function ze(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Db(Object(r), !0).forEach(function(n) {
      NN(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Db(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function NN(e, t, r) {
  return t = DN(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function DN(e) {
  var t = qN(e, "string");
  return di(t) == "symbol" ? t : t + "";
}
function qN(e, t) {
  if (di(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (di(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function LN(e, t, r, n, i) {
  for (var a = (n || []).slice(), o = a.length, u = t.start, s = t.end, c = function(d) {
    var p = a[d], y, v = function() {
      return y === void 0 && (y = r(p, d)), y;
    };
    if (d === o - 1) {
      var h = e * (p.coordinate + e * v() / 2 - s);
      a[d] = p = ze(ze({}, p), {}, {
        tickCoord: h > 0 ? p.coordinate - h * e : p.coordinate
      });
    } else
      a[d] = p = ze(ze({}, p), {}, {
        tickCoord: p.coordinate
      });
    var g = Na(e, p.tickCoord, v, u, s);
    g && (s = p.tickCoord - e * (v() / 2 + i), a[d] = ze(ze({}, p), {}, {
      isShow: !0
    }));
  }, f = o - 1; f >= 0; f--)
    c(f);
  return a;
}
function BN(e, t, r, n, i, a) {
  var o = (n || []).slice(), u = o.length, s = t.start, c = t.end;
  if (a) {
    var f = n[u - 1], l = r(f, u - 1), d = e * (f.coordinate + e * l / 2 - c);
    o[u - 1] = f = ze(ze({}, f), {}, {
      tickCoord: d > 0 ? f.coordinate - d * e : f.coordinate
    });
    var p = Na(e, f.tickCoord, function() {
      return l;
    }, s, c);
    p && (c = f.tickCoord - e * (l / 2 + i), o[u - 1] = ze(ze({}, f), {}, {
      isShow: !0
    }));
  }
  for (var y = a ? u - 1 : u, v = function(w) {
    var b = o[w], O, m = function() {
      return O === void 0 && (O = r(b, w)), O;
    };
    if (w === 0) {
      var x = e * (b.coordinate - e * m() / 2 - s);
      o[w] = b = ze(ze({}, b), {}, {
        tickCoord: x < 0 ? b.coordinate - x * e : b.coordinate
      });
    } else
      o[w] = b = ze(ze({}, b), {}, {
        tickCoord: b.coordinate
      });
    var _ = Na(e, b.tickCoord, m, s, c);
    _ && (s = b.tickCoord + e * (m() / 2 + i), o[w] = ze(ze({}, b), {}, {
      isShow: !0
    }));
  }, h = 0; h < y; h++)
    v(h);
  return o;
}
function ep(e, t, r) {
  var n = e.tick, i = e.ticks, a = e.viewBox, o = e.minTickGap, u = e.orientation, s = e.interval, c = e.tickFormatter, f = e.unit, l = e.angle;
  if (!i || !i.length || !n)
    return [];
  if (K(s) || It.isSsr)
    return RN(i, typeof s == "number" && K(s) ? s : 0);
  var d = [], p = u === "top" || u === "bottom" ? "width" : "height", y = f && p === "width" ? Mn(f, {
    fontSize: t,
    letterSpacing: r
  }) : {
    width: 0,
    height: 0
  }, v = function(b, O) {
    var m = ue(c) ? c(b.value, O) : b.value;
    return p === "width" ? IN(Mn(m, {
      fontSize: t,
      letterSpacing: r
    }), y, l) : Mn(m, {
      fontSize: t,
      letterSpacing: r
    })[p];
  }, h = i.length >= 2 ? yt(i[1].coordinate - i[0].coordinate) : 1, g = $N(a, h, p);
  return s === "equidistantPreserveStart" ? kN(h, g, v, i, o) : (s === "preserveStart" || s === "preserveStartEnd" ? d = BN(h, g, v, i, o, s === "preserveStartEnd") : d = LN(h, g, v, i, o), d.filter(function(w) {
    return w.isShow;
  }));
}
var FN = ["viewBox"], zN = ["viewBox"], UN = ["ticks"];
function Qr(e) {
  "@babel/helpers - typeof";
  return Qr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, Qr(e);
}
function jr() {
  return jr = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, jr.apply(this, arguments);
}
function qb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function He(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? qb(Object(r), !0).forEach(function(n) {
      tp(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : qb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function wl(e, t) {
  if (e == null) return {};
  var r = WN(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function WN(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function GN(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function Lb(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, eO(n.key), n);
  }
}
function HN(e, t, r) {
  return Lb(e.prototype, t), Lb(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function KN(e, t, r) {
  return t = Da(t), VN(e, Qw() ? Reflect.construct(t, r, Da(e).constructor) : t.apply(e, r));
}
function VN(e, t) {
  if (t && (Qr(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return YN(e);
}
function YN(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function Qw() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (Qw = function() {
    return !!e;
  })();
}
function Da(e) {
  return Da = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Da(e);
}
function XN(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Rf(e, t);
}
function Rf(e, t) {
  return Rf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Rf(e, t);
}
function tp(e, t, r) {
  return t = eO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function eO(e) {
  var t = ZN(e, "string");
  return Qr(t) == "symbol" ? t : t + "";
}
function ZN(e, t) {
  if (Qr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (Qr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var pn = /* @__PURE__ */ function(e) {
  function t(r) {
    var n;
    return GN(this, t), n = KN(this, t, [r]), n.state = {
      fontSize: "",
      letterSpacing: ""
    }, n;
  }
  return XN(t, e), HN(t, [{
    key: "shouldComponentUpdate",
    value: function(n, i) {
      var a = n.viewBox, o = wl(n, FN), u = this.props, s = u.viewBox, c = wl(u, zN);
      return !Mr(a, s) || !Mr(o, c) || !Mr(i, this.state);
    }
  }, {
    key: "componentDidMount",
    value: function() {
      var n = this.layerReference;
      if (n) {
        var i = n.getElementsByClassName("recharts-cartesian-axis-tick-value")[0];
        i && this.setState({
          fontSize: window.getComputedStyle(i).fontSize,
          letterSpacing: window.getComputedStyle(i).letterSpacing
        });
      }
    }
    /**
     * Calculate the coordinates of endpoints in ticks
     * @param  {Object} data The data of a simple tick
     * @return {Object} (x1, y1): The coordinate of endpoint close to tick text
     *  (x2, y2): The coordinate of endpoint close to axis
     */
  }, {
    key: "getTickLineCoord",
    value: function(n) {
      var i = this.props, a = i.x, o = i.y, u = i.width, s = i.height, c = i.orientation, f = i.tickSize, l = i.mirror, d = i.tickMargin, p, y, v, h, g, w, b = l ? -1 : 1, O = n.tickSize || f, m = K(n.tickCoord) ? n.tickCoord : n.coordinate;
      switch (c) {
        case "top":
          p = y = n.coordinate, h = o + +!l * s, v = h - b * O, w = v - b * d, g = m;
          break;
        case "left":
          v = h = n.coordinate, y = a + +!l * u, p = y - b * O, g = p - b * d, w = m;
          break;
        case "right":
          v = h = n.coordinate, y = a + +l * u, p = y + b * O, g = p + b * d, w = m;
          break;
        default:
          p = y = n.coordinate, h = o + +l * s, v = h + b * O, w = v + b * d, g = m;
          break;
      }
      return {
        line: {
          x1: p,
          y1: v,
          x2: y,
          y2: h
        },
        tick: {
          x: g,
          y: w
        }
      };
    }
  }, {
    key: "getTickTextAnchor",
    value: function() {
      var n = this.props, i = n.orientation, a = n.mirror, o;
      switch (i) {
        case "left":
          o = a ? "start" : "end";
          break;
        case "right":
          o = a ? "end" : "start";
          break;
        default:
          o = "middle";
          break;
      }
      return o;
    }
  }, {
    key: "getTickVerticalAnchor",
    value: function() {
      var n = this.props, i = n.orientation, a = n.mirror, o = "end";
      switch (i) {
        case "left":
        case "right":
          o = "middle";
          break;
        case "top":
          o = a ? "start" : "end";
          break;
        default:
          o = a ? "end" : "start";
          break;
      }
      return o;
    }
  }, {
    key: "renderAxisLine",
    value: function() {
      var n = this.props, i = n.x, a = n.y, o = n.width, u = n.height, s = n.orientation, c = n.mirror, f = n.axisLine, l = He(He(He({}, fe(this.props, !1)), fe(f, !1)), {}, {
        fill: "none"
      });
      if (s === "top" || s === "bottom") {
        var d = +(s === "top" && !c || s === "bottom" && c);
        l = He(He({}, l), {}, {
          x1: i,
          y1: a + d * u,
          x2: i + o,
          y2: a + d * u
        });
      } else {
        var p = +(s === "left" && !c || s === "right" && c);
        l = He(He({}, l), {}, {
          x1: i + p * o,
          y1: a,
          x2: i + p * o,
          y2: a + u
        });
      }
      return /* @__PURE__ */ T.createElement("line", jr({}, l, {
        className: pe("recharts-cartesian-axis-line", at(f, "className"))
      }));
    }
  }, {
    key: "renderTicks",
    value: (
      /**
       * render the ticks
       * @param {Array} ticks The ticks to actually render (overrides what was passed in props)
       * @param {string} fontSize Fontsize to consider for tick spacing
       * @param {string} letterSpacing Letterspacing to consider for tick spacing
       * @return {ReactComponent} renderedTicks
       */
      function(n, i, a) {
        var o = this, u = this.props, s = u.tickLine, c = u.stroke, f = u.tick, l = u.tickFormatter, d = u.unit, p = ep(He(He({}, this.props), {}, {
          ticks: n
        }), i, a), y = this.getTickTextAnchor(), v = this.getTickVerticalAnchor(), h = fe(this.props, !1), g = fe(f, !1), w = He(He({}, h), {}, {
          fill: "none"
        }, fe(s, !1)), b = p.map(function(O, m) {
          var x = o.getTickLineCoord(O), _ = x.line, P = x.tick, E = He(He(He(He({
            textAnchor: y,
            verticalAnchor: v
          }, h), {}, {
            stroke: "none",
            fill: c
          }, g), P), {}, {
            index: m,
            payload: O,
            visibleTicksCount: p.length,
            tickFormatter: l
          });
          return /* @__PURE__ */ T.createElement(je, jr({
            className: "recharts-cartesian-axis-tick",
            key: "tick-".concat(O.value, "-").concat(O.coordinate, "-").concat(O.tickCoord)
          }, Hi(o.props, O, m)), s && /* @__PURE__ */ T.createElement("line", jr({}, w, _, {
            className: pe("recharts-cartesian-axis-tick-line", at(s, "className"))
          })), f && t.renderTickItem(f, E, "".concat(ue(l) ? l(O.value, m) : O.value).concat(d || "")));
        });
        return /* @__PURE__ */ T.createElement("g", {
          className: "recharts-cartesian-axis-ticks"
        }, b);
      }
    )
  }, {
    key: "render",
    value: function() {
      var n = this, i = this.props, a = i.axisLine, o = i.width, u = i.height, s = i.ticksGenerator, c = i.className, f = i.hide;
      if (f)
        return null;
      var l = this.props, d = l.ticks, p = wl(l, UN), y = d;
      return ue(s) && (y = d && d.length > 0 ? s(this.props) : s(p)), o <= 0 || u <= 0 || !y || !y.length ? null : /* @__PURE__ */ T.createElement(je, {
        className: pe("recharts-cartesian-axis", c),
        ref: function(h) {
          n.layerReference = h;
        }
      }, a && this.renderAxisLine(), this.renderTicks(y, this.state.fontSize, this.state.letterSpacing), Ue.renderCallByParent(this.props));
    }
  }], [{
    key: "renderTickItem",
    value: function(n, i, a) {
      var o;
      return /* @__PURE__ */ T.isValidElement(n) ? o = /* @__PURE__ */ T.cloneElement(n, i) : ue(n) ? o = n(i) : o = /* @__PURE__ */ T.createElement(na, jr({}, i, {
        className: "recharts-cartesian-axis-tick-value"
      }), a), o;
    }
  }]);
}(u0);
tp(pn, "displayName", "CartesianAxis");
tp(pn, "defaultProps", {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  viewBox: {
    x: 0,
    y: 0,
    width: 0,
    height: 0
  },
  // The orientation of axis
  orientation: "bottom",
  // The ticks
  ticks: [],
  stroke: "#666",
  tickLine: !0,
  axisLine: !0,
  tick: !0,
  mirror: !1,
  minTickGap: 5,
  // The width or height of tick
  tickSize: 6,
  tickMargin: 2,
  interval: "preserveEnd"
});
var JN = ["x1", "y1", "x2", "y2", "key"], QN = ["offset"];
function yr(e) {
  "@babel/helpers - typeof";
  return yr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, yr(e);
}
function Bb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function We(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Bb(Object(r), !0).forEach(function(n) {
      eD(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Bb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function eD(e, t, r) {
  return t = tD(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function tD(e) {
  var t = rD(e, "string");
  return yr(t) == "symbol" ? t : t + "";
}
function rD(e, t) {
  if (yr(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (yr(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function cr() {
  return cr = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, cr.apply(this, arguments);
}
function Fb(e, t) {
  if (e == null) return {};
  var r = nD(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function nD(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
var iD = function(t) {
  var r = t.fill;
  if (!r || r === "none")
    return null;
  var n = t.fillOpacity, i = t.x, a = t.y, o = t.width, u = t.height, s = t.ry;
  return /* @__PURE__ */ T.createElement("rect", {
    x: i,
    y: a,
    ry: s,
    width: o,
    height: u,
    stroke: "none",
    fill: r,
    fillOpacity: n,
    className: "recharts-cartesian-grid-bg"
  });
};
function tO(e, t) {
  var r;
  if (/* @__PURE__ */ T.isValidElement(e))
    r = /* @__PURE__ */ T.cloneElement(e, t);
  else if (ue(e))
    r = e(t);
  else {
    var n = t.x1, i = t.y1, a = t.x2, o = t.y2, u = t.key, s = Fb(t, JN), c = fe(s, !1);
    c.offset;
    var f = Fb(c, QN);
    r = /* @__PURE__ */ T.createElement("line", cr({}, f, {
      x1: n,
      y1: i,
      x2: a,
      y2: o,
      fill: "none",
      key: u
    }));
  }
  return r;
}
function aD(e) {
  var t = e.x, r = e.width, n = e.horizontal, i = n === void 0 ? !0 : n, a = e.horizontalPoints;
  if (!i || !a || !a.length)
    return null;
  var o = a.map(function(u, s) {
    var c = We(We({}, e), {}, {
      x1: t,
      y1: u,
      x2: t + r,
      y2: u,
      key: "line-".concat(s),
      index: s
    });
    return tO(i, c);
  });
  return /* @__PURE__ */ T.createElement("g", {
    className: "recharts-cartesian-grid-horizontal"
  }, o);
}
function oD(e) {
  var t = e.y, r = e.height, n = e.vertical, i = n === void 0 ? !0 : n, a = e.verticalPoints;
  if (!i || !a || !a.length)
    return null;
  var o = a.map(function(u, s) {
    var c = We(We({}, e), {}, {
      x1: u,
      y1: t,
      x2: u,
      y2: t + r,
      key: "line-".concat(s),
      index: s
    });
    return tO(i, c);
  });
  return /* @__PURE__ */ T.createElement("g", {
    className: "recharts-cartesian-grid-vertical"
  }, o);
}
function uD(e) {
  var t = e.horizontalFill, r = e.fillOpacity, n = e.x, i = e.y, a = e.width, o = e.height, u = e.horizontalPoints, s = e.horizontal, c = s === void 0 ? !0 : s;
  if (!c || !t || !t.length)
    return null;
  var f = u.map(function(d) {
    return Math.round(d + i - i);
  }).sort(function(d, p) {
    return d - p;
  });
  i !== f[0] && f.unshift(0);
  var l = f.map(function(d, p) {
    var y = !f[p + 1], v = y ? i + o - d : f[p + 1] - d;
    if (v <= 0)
      return null;
    var h = p % t.length;
    return /* @__PURE__ */ T.createElement("rect", {
      key: "react-".concat(p),
      y: d,
      x: n,
      height: v,
      width: a,
      stroke: "none",
      fill: t[h],
      fillOpacity: r,
      className: "recharts-cartesian-grid-bg"
    });
  });
  return /* @__PURE__ */ T.createElement("g", {
    className: "recharts-cartesian-gridstripes-horizontal"
  }, l);
}
function sD(e) {
  var t = e.vertical, r = t === void 0 ? !0 : t, n = e.verticalFill, i = e.fillOpacity, a = e.x, o = e.y, u = e.width, s = e.height, c = e.verticalPoints;
  if (!r || !n || !n.length)
    return null;
  var f = c.map(function(d) {
    return Math.round(d + a - a);
  }).sort(function(d, p) {
    return d - p;
  });
  a !== f[0] && f.unshift(0);
  var l = f.map(function(d, p) {
    var y = !f[p + 1], v = y ? a + u - d : f[p + 1] - d;
    if (v <= 0)
      return null;
    var h = p % n.length;
    return /* @__PURE__ */ T.createElement("rect", {
      key: "react-".concat(p),
      x: d,
      y: o,
      width: v,
      height: s,
      stroke: "none",
      fill: n[h],
      fillOpacity: i,
      className: "recharts-cartesian-grid-bg"
    });
  });
  return /* @__PURE__ */ T.createElement("g", {
    className: "recharts-cartesian-gridstripes-vertical"
  }, l);
}
var cD = function(t, r) {
  var n = t.xAxis, i = t.width, a = t.height, o = t.offset;
  return dw(ep(We(We(We({}, pn.defaultProps), n), {}, {
    ticks: Mt(n, !0),
    viewBox: {
      x: 0,
      y: 0,
      width: i,
      height: a
    }
  })), o.left, o.left + o.width, r);
}, lD = function(t, r) {
  var n = t.yAxis, i = t.width, a = t.height, o = t.offset;
  return dw(ep(We(We(We({}, pn.defaultProps), n), {}, {
    ticks: Mt(n, !0),
    viewBox: {
      x: 0,
      y: 0,
      width: i,
      height: a
    }
  })), o.top, o.top + o.height, r);
}, Pr = {
  horizontal: !0,
  vertical: !0,
  // The ordinates of horizontal grid lines
  horizontalPoints: [],
  // The abscissas of vertical grid lines
  verticalPoints: [],
  stroke: "#ccc",
  fill: "none",
  // The fill of colors of grid lines
  verticalFill: [],
  horizontalFill: []
};
function rO(e) {
  var t, r, n, i, a, o, u = Xd(), s = Zd(), c = Zk(), f = We(We({}, e), {}, {
    stroke: (t = e.stroke) !== null && t !== void 0 ? t : Pr.stroke,
    fill: (r = e.fill) !== null && r !== void 0 ? r : Pr.fill,
    horizontal: (n = e.horizontal) !== null && n !== void 0 ? n : Pr.horizontal,
    horizontalFill: (i = e.horizontalFill) !== null && i !== void 0 ? i : Pr.horizontalFill,
    vertical: (a = e.vertical) !== null && a !== void 0 ? a : Pr.vertical,
    verticalFill: (o = e.verticalFill) !== null && o !== void 0 ? o : Pr.verticalFill,
    x: K(e.x) ? e.x : c.left,
    y: K(e.y) ? e.y : c.top,
    width: K(e.width) ? e.width : c.width,
    height: K(e.height) ? e.height : c.height
  }), l = f.x, d = f.y, p = f.width, y = f.height, v = f.syncWithTicks, h = f.horizontalValues, g = f.verticalValues, w = Vk(), b = Yk();
  if (!K(p) || p <= 0 || !K(y) || y <= 0 || !K(l) || l !== +l || !K(d) || d !== +d)
    return null;
  var O = f.verticalCoordinatesGenerator || cD, m = f.horizontalCoordinatesGenerator || lD, x = f.horizontalPoints, _ = f.verticalPoints;
  if ((!x || !x.length) && ue(m)) {
    var P = h && h.length, E = m({
      yAxis: b ? We(We({}, b), {}, {
        ticks: P ? h : b.ticks
      }) : void 0,
      width: u,
      height: s,
      offset: c
    }, P ? !0 : v);
    kr(Array.isArray(E), "horizontalCoordinatesGenerator should return Array but instead it returned [".concat(yr(E), "]")), Array.isArray(E) && (x = E);
  }
  if ((!_ || !_.length) && ue(O)) {
    var I = g && g.length, S = O({
      xAxis: w ? We(We({}, w), {}, {
        ticks: I ? g : w.ticks
      }) : void 0,
      width: u,
      height: s,
      offset: c
    }, I ? !0 : v);
    kr(Array.isArray(S), "verticalCoordinatesGenerator should return Array but instead it returned [".concat(yr(S), "]")), Array.isArray(S) && (_ = S);
  }
  return /* @__PURE__ */ T.createElement("g", {
    className: "recharts-cartesian-grid"
  }, /* @__PURE__ */ T.createElement(iD, {
    fill: f.fill,
    fillOpacity: f.fillOpacity,
    x: f.x,
    y: f.y,
    width: f.width,
    height: f.height,
    ry: f.ry
  }), /* @__PURE__ */ T.createElement(aD, cr({}, f, {
    offset: c,
    horizontalPoints: x,
    xAxis: w,
    yAxis: b
  })), /* @__PURE__ */ T.createElement(oD, cr({}, f, {
    offset: c,
    verticalPoints: _,
    xAxis: w,
    yAxis: b
  })), /* @__PURE__ */ T.createElement(uD, cr({}, f, {
    horizontalPoints: x
  })), /* @__PURE__ */ T.createElement(sD, cr({}, f, {
    verticalPoints: _
  })));
}
rO.displayName = "CartesianGrid";
var fD = ["type", "layout", "connectNulls", "ref"], dD = ["key"];
function en(e) {
  "@babel/helpers - typeof";
  return en = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, en(e);
}
function zb(e, t) {
  if (e == null) return {};
  var r = pD(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function pD(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function kn() {
  return kn = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, kn.apply(this, arguments);
}
function Ub(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Ze(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Ub(Object(r), !0).forEach(function(n) {
      pt(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ub(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function Ar(e) {
  return mD(e) || yD(e) || vD(e) || hD();
}
function hD() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function vD(e, t) {
  if (e) {
    if (typeof e == "string") return kf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return kf(e, t);
  }
}
function yD(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function mD(e) {
  if (Array.isArray(e)) return kf(e);
}
function kf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function gD(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function Wb(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, iO(n.key), n);
  }
}
function bD(e, t, r) {
  return Wb(e.prototype, t), Wb(e, r), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function xD(e, t, r) {
  return t = qa(t), wD(e, nO() ? Reflect.construct(t, r, qa(e).constructor) : t.apply(e, r));
}
function wD(e, t) {
  if (t && (en(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return OD(e);
}
function OD(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function nO() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (nO = function() {
    return !!e;
  })();
}
function qa(e) {
  return qa = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, qa(e);
}
function _D(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Nf(e, t);
}
function Nf(e, t) {
  return Nf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Nf(e, t);
}
function pt(e, t, r) {
  return t = iO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function iO(e) {
  var t = SD(e, "string");
  return en(t) == "symbol" ? t : t + "";
}
function SD(e, t) {
  if (en(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (en(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var tn = /* @__PURE__ */ function(e) {
  function t() {
    var r;
    gD(this, t);
    for (var n = arguments.length, i = new Array(n), a = 0; a < n; a++)
      i[a] = arguments[a];
    return r = xD(this, t, [].concat(i)), pt(r, "state", {
      isAnimationFinished: !0,
      totalLength: 0
    }), pt(r, "generateSimpleStrokeDasharray", function(o, u) {
      return "".concat(u, "px ").concat(o - u, "px");
    }), pt(r, "getStrokeDasharray", function(o, u, s) {
      var c = s.reduce(function(g, w) {
        return g + w;
      });
      if (!c)
        return r.generateSimpleStrokeDasharray(u, o);
      for (var f = Math.floor(o / c), l = o % c, d = u - o, p = [], y = 0, v = 0; y < s.length; v += s[y], ++y)
        if (v + s[y] > l) {
          p = [].concat(Ar(s.slice(0, y)), [l - v]);
          break;
        }
      var h = p.length % 2 === 0 ? [0, d] : [d];
      return [].concat(Ar(t.repeat(s, f)), Ar(p), h).map(function(g) {
        return "".concat(g, "px");
      }).join(", ");
    }), pt(r, "id", gi("recharts-line-")), pt(r, "pathRef", function(o) {
      r.mainCurve = o;
    }), pt(r, "handleAnimationEnd", function() {
      r.setState({
        isAnimationFinished: !0
      }), r.props.onAnimationEnd && r.props.onAnimationEnd();
    }), pt(r, "handleAnimationStart", function() {
      r.setState({
        isAnimationFinished: !1
      }), r.props.onAnimationStart && r.props.onAnimationStart();
    }), r;
  }
  return _D(t, e), bD(t, [{
    key: "componentDidMount",
    value: function() {
      if (this.props.isAnimationActive) {
        var n = this.getTotalLength();
        this.setState({
          totalLength: n
        });
      }
    }
  }, {
    key: "componentDidUpdate",
    value: function() {
      if (this.props.isAnimationActive) {
        var n = this.getTotalLength();
        n !== this.state.totalLength && this.setState({
          totalLength: n
        });
      }
    }
  }, {
    key: "getTotalLength",
    value: function() {
      var n = this.mainCurve;
      try {
        return n && n.getTotalLength && n.getTotalLength() || 0;
      } catch {
        return 0;
      }
    }
  }, {
    key: "renderErrorBar",
    value: function(n, i) {
      if (this.props.isAnimationActive && !this.state.isAnimationFinished)
        return null;
      var a = this.props, o = a.points, u = a.xAxis, s = a.yAxis, c = a.layout, f = a.children, l = ot(f, Si);
      if (!l)
        return null;
      var d = function(v, h) {
        return {
          x: v.x,
          y: v.y,
          value: v.value,
          errorVal: tt(v.payload, h)
        };
      }, p = {
        clipPath: n ? "url(#clipPath-".concat(i, ")") : null
      };
      return /* @__PURE__ */ T.createElement(je, p, l.map(function(y) {
        return /* @__PURE__ */ T.cloneElement(y, {
          key: "bar-".concat(y.props.dataKey),
          data: o,
          xAxis: u,
          yAxis: s,
          layout: c,
          dataPointFormatter: d
        });
      }));
    }
  }, {
    key: "renderDots",
    value: function(n, i, a) {
      var o = this.props.isAnimationActive;
      if (o && !this.state.isAnimationFinished)
        return null;
      var u = this.props, s = u.dot, c = u.points, f = u.dataKey, l = fe(this.props, !1), d = fe(s, !0), p = c.map(function(v, h) {
        var g = Ze(Ze(Ze({
          key: "dot-".concat(h),
          r: 3
        }, l), d), {}, {
          value: v.value,
          dataKey: f,
          cx: v.x,
          cy: v.y,
          index: h,
          payload: v.payload
        });
        return t.renderDotItem(s, g);
      }), y = {
        clipPath: n ? "url(#clipPath-".concat(i ? "" : "dots-").concat(a, ")") : null
      };
      return /* @__PURE__ */ T.createElement(je, kn({
        className: "recharts-line-dots",
        key: "dots"
      }, y), p);
    }
  }, {
    key: "renderCurveStatically",
    value: function(n, i, a, o) {
      var u = this.props, s = u.type, c = u.layout, f = u.connectNulls;
      u.ref;
      var l = zb(u, fD), d = Ze(Ze(Ze({}, fe(l, !0)), {}, {
        fill: "none",
        className: "recharts-line-curve",
        clipPath: i ? "url(#clipPath-".concat(a, ")") : null,
        points: n
      }, o), {}, {
        type: s,
        layout: c,
        connectNulls: f
      });
      return /* @__PURE__ */ T.createElement(pf, kn({}, d, {
        pathRef: this.pathRef
      }));
    }
  }, {
    key: "renderCurveWithAnimation",
    value: function(n, i) {
      var a = this, o = this.props, u = o.points, s = o.strokeDasharray, c = o.isAnimationActive, f = o.animationBegin, l = o.animationDuration, d = o.animationEasing, p = o.animationId, y = o.animateNewValues, v = o.width, h = o.height, g = this.state, w = g.prevPoints, b = g.totalLength;
      return /* @__PURE__ */ T.createElement(Dt, {
        begin: f,
        duration: l,
        isActive: c,
        easing: d,
        from: {
          t: 0
        },
        to: {
          t: 1
        },
        key: "line-".concat(p),
        onAnimationEnd: this.handleAnimationEnd,
        onAnimationStart: this.handleAnimationStart
      }, function(O) {
        var m = O.t;
        if (w) {
          var x = w.length / u.length, _ = u.map(function(j, M) {
            var R = Math.floor(M * x);
            if (w[R]) {
              var k = w[R], q = ht(k.x, j.x), L = ht(k.y, j.y);
              return Ze(Ze({}, j), {}, {
                x: q(m),
                y: L(m)
              });
            }
            if (y) {
              var U = ht(v * 2, j.x), z = ht(h / 2, j.y);
              return Ze(Ze({}, j), {}, {
                x: U(m),
                y: z(m)
              });
            }
            return Ze(Ze({}, j), {}, {
              x: j.x,
              y: j.y
            });
          });
          return a.renderCurveStatically(_, n, i);
        }
        var P = ht(0, b), E = P(m), I;
        if (s) {
          var S = "".concat(s).split(/[,\s]+/gim).map(function(j) {
            return parseFloat(j);
          });
          I = a.getStrokeDasharray(E, b, S);
        } else
          I = a.generateSimpleStrokeDasharray(b, E);
        return a.renderCurveStatically(u, n, i, {
          strokeDasharray: I
        });
      });
    }
  }, {
    key: "renderCurve",
    value: function(n, i) {
      var a = this.props, o = a.points, u = a.isAnimationActive, s = this.state, c = s.prevPoints, f = s.totalLength;
      return u && o && o.length && (!c && f > 0 || !co(c, o)) ? this.renderCurveWithAnimation(n, i) : this.renderCurveStatically(o, n, i);
    }
  }, {
    key: "render",
    value: function() {
      var n, i = this.props, a = i.hide, o = i.dot, u = i.points, s = i.className, c = i.xAxis, f = i.yAxis, l = i.top, d = i.left, p = i.width, y = i.height, v = i.isAnimationActive, h = i.id;
      if (a || !u || !u.length)
        return null;
      var g = this.state.isAnimationFinished, w = u.length === 1, b = pe("recharts-line", s), O = c && c.allowDataOverflow, m = f && f.allowDataOverflow, x = O || m, _ = ce(h) ? this.id : h, P = (n = fe(o, !1)) !== null && n !== void 0 ? n : {
        r: 3,
        strokeWidth: 2
      }, E = P.r, I = E === void 0 ? 3 : E, S = P.strokeWidth, j = S === void 0 ? 2 : S, M = U_(o) ? o : {}, R = M.clipDot, k = R === void 0 ? !0 : R, q = I * 2 + j;
      return /* @__PURE__ */ T.createElement(je, {
        className: b
      }, O || m ? /* @__PURE__ */ T.createElement("defs", null, /* @__PURE__ */ T.createElement("clipPath", {
        id: "clipPath-".concat(_)
      }, /* @__PURE__ */ T.createElement("rect", {
        x: O ? d : d - p / 2,
        y: m ? l : l - y / 2,
        width: O ? p : p * 2,
        height: m ? y : y * 2
      })), !k && /* @__PURE__ */ T.createElement("clipPath", {
        id: "clipPath-dots-".concat(_)
      }, /* @__PURE__ */ T.createElement("rect", {
        x: d - q / 2,
        y: l - q / 2,
        width: p + q,
        height: y + q
      }))) : null, !w && this.renderCurve(x, _), this.renderErrorBar(x, _), (w || o) && this.renderDots(x, k, _), (!v || g) && Vt.renderCallByParent(this.props, u));
    }
  }], [{
    key: "getDerivedStateFromProps",
    value: function(n, i) {
      return n.animationId !== i.prevAnimationId ? {
        prevAnimationId: n.animationId,
        curPoints: n.points,
        prevPoints: i.curPoints
      } : n.points !== i.curPoints ? {
        curPoints: n.points
      } : null;
    }
  }, {
    key: "repeat",
    value: function(n, i) {
      for (var a = n.length % 2 !== 0 ? [].concat(Ar(n), [0]) : n, o = [], u = 0; u < i; ++u)
        o = [].concat(Ar(o), Ar(a));
      return o;
    }
  }, {
    key: "renderDotItem",
    value: function(n, i) {
      var a;
      if (/* @__PURE__ */ T.isValidElement(n))
        a = /* @__PURE__ */ T.cloneElement(n, i);
      else if (ue(n))
        a = n(i);
      else {
        var o = i.key, u = zb(i, dD), s = pe("recharts-line-dot", typeof n != "boolean" ? n.className : "");
        a = /* @__PURE__ */ T.createElement(Hd, kn({
          key: o
        }, u, {
          className: s
        }));
      }
      return a;
    }
  }]);
}(Xt);
pt(tn, "displayName", "Line");
pt(tn, "defaultProps", {
  xAxisId: 0,
  yAxisId: 0,
  connectNulls: !1,
  activeDot: !0,
  dot: !0,
  legendType: "line",
  stroke: "#3182bd",
  strokeWidth: 1,
  fill: "#fff",
  points: [],
  isAnimationActive: !It.isSsr,
  animateNewValues: !0,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: "ease",
  hide: !1,
  label: !1
});
pt(tn, "getComposedData", function(e) {
  var t = e.props, r = e.xAxis, n = e.yAxis, i = e.xAxisTicks, a = e.yAxisTicks, o = e.dataKey, u = e.bandSize, s = e.displayedData, c = e.offset, f = t.layout, l = s.map(function(d, p) {
    var y = tt(d, o);
    return f === "horizontal" ? {
      x: Zm({
        axis: r,
        ticks: i,
        bandSize: u,
        entry: d,
        index: p
      }),
      y: ce(y) ? null : n.scale(y),
      value: y,
      payload: d
    } : {
      x: ce(y) ? null : r.scale(y),
      y: Zm({
        axis: n,
        ticks: a,
        bandSize: u,
        entry: d,
        index: p
      }),
      value: y,
      payload: d
    };
  });
  return Ze({
    points: l,
    layout: f
  }, c);
});
function rn(e) {
  "@babel/helpers - typeof";
  return rn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, rn(e);
}
function PD(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function AD(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, uO(n.key), n);
  }
}
function ED(e, t, r) {
  return AD(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function TD(e, t, r) {
  return t = La(t), jD(e, aO() ? Reflect.construct(t, r || [], La(e).constructor) : t.apply(e, r));
}
function jD(e, t) {
  if (t && (rn(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return CD(e);
}
function CD(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function aO() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (aO = function() {
    return !!e;
  })();
}
function La(e) {
  return La = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, La(e);
}
function MD(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Df(e, t);
}
function Df(e, t) {
  return Df = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Df(e, t);
}
function oO(e, t, r) {
  return t = uO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function uO(e) {
  var t = ID(e, "string");
  return rn(t) == "symbol" ? t : t + "";
}
function ID(e, t) {
  if (rn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (rn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function qf() {
  return qf = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, qf.apply(this, arguments);
}
function $D(e) {
  var t = e.xAxisId, r = Xd(), n = Zd(), i = Ww(t);
  return i == null ? null : (
    // @ts-expect-error the axisOptions type is not exactly what CartesianAxis is expecting.
    /* @__PURE__ */ T.createElement(pn, qf({}, i, {
      className: pe("recharts-".concat(i.axisType, " ").concat(i.axisType), i.className),
      viewBox: {
        x: 0,
        y: 0,
        width: r,
        height: n
      },
      ticksGenerator: function(o) {
        return Mt(o, !0);
      }
    }))
  );
}
var xo = /* @__PURE__ */ function(e) {
  function t() {
    return PD(this, t), TD(this, t, arguments);
  }
  return MD(t, e), ED(t, [{
    key: "render",
    value: function() {
      return /* @__PURE__ */ T.createElement($D, this.props);
    }
  }]);
}(T.Component);
oO(xo, "displayName", "XAxis");
oO(xo, "defaultProps", {
  allowDecimals: !0,
  hide: !1,
  orientation: "bottom",
  width: 0,
  height: 30,
  mirror: !1,
  xAxisId: 0,
  tickCount: 5,
  type: "category",
  padding: {
    left: 0,
    right: 0
  },
  allowDataOverflow: !1,
  scale: "auto",
  reversed: !1,
  allowDuplicatedCategory: !0
});
function nn(e) {
  "@babel/helpers - typeof";
  return nn = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, nn(e);
}
function RD(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function kD(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, lO(n.key), n);
  }
}
function ND(e, t, r) {
  return kD(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function DD(e, t, r) {
  return t = Ba(t), qD(e, sO() ? Reflect.construct(t, r || [], Ba(e).constructor) : t.apply(e, r));
}
function qD(e, t) {
  if (t && (nn(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return LD(e);
}
function LD(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function sO() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (sO = function() {
    return !!e;
  })();
}
function Ba(e) {
  return Ba = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Ba(e);
}
function BD(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Lf(e, t);
}
function Lf(e, t) {
  return Lf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Lf(e, t);
}
function cO(e, t, r) {
  return t = lO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function lO(e) {
  var t = FD(e, "string");
  return nn(t) == "symbol" ? t : t + "";
}
function FD(e, t) {
  if (nn(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (nn(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function Bf() {
  return Bf = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Bf.apply(this, arguments);
}
var zD = function(t) {
  var r = t.yAxisId, n = Xd(), i = Zd(), a = Gw(r);
  return a == null ? null : (
    // @ts-expect-error the axisOptions type is not exactly what CartesianAxis is expecting.
    /* @__PURE__ */ T.createElement(pn, Bf({}, a, {
      className: pe("recharts-".concat(a.axisType, " ").concat(a.axisType), a.className),
      viewBox: {
        x: 0,
        y: 0,
        width: n,
        height: i
      },
      ticksGenerator: function(u) {
        return Mt(u, !0);
      }
    }))
  );
}, wo = /* @__PURE__ */ function(e) {
  function t() {
    return RD(this, t), DD(this, t, arguments);
  }
  return BD(t, e), ND(t, [{
    key: "render",
    value: function() {
      return /* @__PURE__ */ T.createElement(zD, this.props);
    }
  }]);
}(T.Component);
cO(wo, "displayName", "YAxis");
cO(wo, "defaultProps", {
  allowDuplicatedCategory: !0,
  allowDecimals: !0,
  hide: !1,
  orientation: "left",
  width: 60,
  height: 0,
  mirror: !1,
  yAxisId: 0,
  tickCount: 5,
  type: "number",
  padding: {
    top: 0,
    bottom: 0
  },
  allowDataOverflow: !1,
  scale: "auto",
  reversed: !1
});
function Gb(e) {
  return HD(e) || GD(e) || WD(e) || UD();
}
function UD() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function WD(e, t) {
  if (e) {
    if (typeof e == "string") return Ff(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Ff(e, t);
  }
}
function GD(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function HD(e) {
  if (Array.isArray(e)) return Ff(e);
}
function Ff(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
var zf = function(t, r, n, i, a) {
  var o = ot(t, Qd), u = ot(t, mo), s = [].concat(Gb(o), Gb(u)), c = ot(t, bo), f = "".concat(i, "Id"), l = i[0], d = r;
  if (s.length && (d = s.reduce(function(v, h) {
    if (h.props[f] === n && Ot(h.props, "extendDomain") && K(h.props[l])) {
      var g = h.props[l];
      return [Math.min(v[0], g), Math.max(v[1], g)];
    }
    return v;
  }, d)), c.length) {
    var p = "".concat(l, "1"), y = "".concat(l, "2");
    d = c.reduce(function(v, h) {
      if (h.props[f] === n && Ot(h.props, "extendDomain") && K(h.props[p]) && K(h.props[y])) {
        var g = h.props[p], w = h.props[y];
        return [Math.min(v[0], g, w), Math.max(v[1], g, w)];
      }
      return v;
    }, d);
  }
  return a && a.length && (d = a.reduce(function(v, h) {
    return K(h) ? [Math.min(v[0], h), Math.max(v[1], h)] : v;
  }, d)), d;
}, Ol = { exports: {} }, Hb;
function KD() {
  return Hb || (Hb = 1, function(e) {
    var t = Object.prototype.hasOwnProperty, r = "~";
    function n() {
    }
    Object.create && (n.prototype = /* @__PURE__ */ Object.create(null), new n().__proto__ || (r = !1));
    function i(s, c, f) {
      this.fn = s, this.context = c, this.once = f || !1;
    }
    function a(s, c, f, l, d) {
      if (typeof f != "function")
        throw new TypeError("The listener must be a function");
      var p = new i(f, l || s, d), y = r ? r + c : c;
      return s._events[y] ? s._events[y].fn ? s._events[y] = [s._events[y], p] : s._events[y].push(p) : (s._events[y] = p, s._eventsCount++), s;
    }
    function o(s, c) {
      --s._eventsCount === 0 ? s._events = new n() : delete s._events[c];
    }
    function u() {
      this._events = new n(), this._eventsCount = 0;
    }
    u.prototype.eventNames = function() {
      var c = [], f, l;
      if (this._eventsCount === 0) return c;
      for (l in f = this._events)
        t.call(f, l) && c.push(r ? l.slice(1) : l);
      return Object.getOwnPropertySymbols ? c.concat(Object.getOwnPropertySymbols(f)) : c;
    }, u.prototype.listeners = function(c) {
      var f = r ? r + c : c, l = this._events[f];
      if (!l) return [];
      if (l.fn) return [l.fn];
      for (var d = 0, p = l.length, y = new Array(p); d < p; d++)
        y[d] = l[d].fn;
      return y;
    }, u.prototype.listenerCount = function(c) {
      var f = r ? r + c : c, l = this._events[f];
      return l ? l.fn ? 1 : l.length : 0;
    }, u.prototype.emit = function(c, f, l, d, p, y) {
      var v = r ? r + c : c;
      if (!this._events[v]) return !1;
      var h = this._events[v], g = arguments.length, w, b;
      if (h.fn) {
        switch (h.once && this.removeListener(c, h.fn, void 0, !0), g) {
          case 1:
            return h.fn.call(h.context), !0;
          case 2:
            return h.fn.call(h.context, f), !0;
          case 3:
            return h.fn.call(h.context, f, l), !0;
          case 4:
            return h.fn.call(h.context, f, l, d), !0;
          case 5:
            return h.fn.call(h.context, f, l, d, p), !0;
          case 6:
            return h.fn.call(h.context, f, l, d, p, y), !0;
        }
        for (b = 1, w = new Array(g - 1); b < g; b++)
          w[b - 1] = arguments[b];
        h.fn.apply(h.context, w);
      } else {
        var O = h.length, m;
        for (b = 0; b < O; b++)
          switch (h[b].once && this.removeListener(c, h[b].fn, void 0, !0), g) {
            case 1:
              h[b].fn.call(h[b].context);
              break;
            case 2:
              h[b].fn.call(h[b].context, f);
              break;
            case 3:
              h[b].fn.call(h[b].context, f, l);
              break;
            case 4:
              h[b].fn.call(h[b].context, f, l, d);
              break;
            default:
              if (!w) for (m = 1, w = new Array(g - 1); m < g; m++)
                w[m - 1] = arguments[m];
              h[b].fn.apply(h[b].context, w);
          }
      }
      return !0;
    }, u.prototype.on = function(c, f, l) {
      return a(this, c, f, l, !1);
    }, u.prototype.once = function(c, f, l) {
      return a(this, c, f, l, !0);
    }, u.prototype.removeListener = function(c, f, l, d) {
      var p = r ? r + c : c;
      if (!this._events[p]) return this;
      if (!f)
        return o(this, p), this;
      var y = this._events[p];
      if (y.fn)
        y.fn === f && (!d || y.once) && (!l || y.context === l) && o(this, p);
      else {
        for (var v = 0, h = [], g = y.length; v < g; v++)
          (y[v].fn !== f || d && !y[v].once || l && y[v].context !== l) && h.push(y[v]);
        h.length ? this._events[p] = h.length === 1 ? h[0] : h : o(this, p);
      }
      return this;
    }, u.prototype.removeAllListeners = function(c) {
      var f;
      return c ? (f = r ? r + c : c, this._events[f] && o(this, f)) : (this._events = new n(), this._eventsCount = 0), this;
    }, u.prototype.off = u.prototype.removeListener, u.prototype.addListener = u.prototype.on, u.prefixed = r, u.EventEmitter = u, e.exports = u;
  }(Ol)), Ol.exports;
}
var VD = KD();
const YD = /* @__PURE__ */ Pe(VD);
var _l = new YD(), Sl = "recharts.syncMouseEvents";
function pi(e) {
  "@babel/helpers - typeof";
  return pi = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, pi(e);
}
function XD(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function ZD(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, fO(n.key), n);
  }
}
function JD(e, t, r) {
  return ZD(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function Pl(e, t, r) {
  return t = fO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function fO(e) {
  var t = QD(e, "string");
  return pi(t) == "symbol" ? t : t + "";
}
function QD(e, t) {
  if (pi(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t);
    if (pi(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return String(e);
}
var eq = /* @__PURE__ */ function() {
  function e() {
    XD(this, e), Pl(this, "activeIndex", 0), Pl(this, "coordinateList", []), Pl(this, "layout", "horizontal");
  }
  return JD(e, [{
    key: "setDetails",
    value: function(r) {
      var n, i = r.coordinateList, a = i === void 0 ? null : i, o = r.container, u = o === void 0 ? null : o, s = r.layout, c = s === void 0 ? null : s, f = r.offset, l = f === void 0 ? null : f, d = r.mouseHandlerCallback, p = d === void 0 ? null : d;
      this.coordinateList = (n = a ?? this.coordinateList) !== null && n !== void 0 ? n : [], this.container = u ?? this.container, this.layout = c ?? this.layout, this.offset = l ?? this.offset, this.mouseHandlerCallback = p ?? this.mouseHandlerCallback, this.activeIndex = Math.min(Math.max(this.activeIndex, 0), this.coordinateList.length - 1);
    }
  }, {
    key: "focus",
    value: function() {
      this.spoofMouse();
    }
  }, {
    key: "keyboardEvent",
    value: function(r) {
      if (this.coordinateList.length !== 0)
        switch (r.key) {
          case "ArrowRight": {
            if (this.layout !== "horizontal")
              return;
            this.activeIndex = Math.min(this.activeIndex + 1, this.coordinateList.length - 1), this.spoofMouse();
            break;
          }
          case "ArrowLeft": {
            if (this.layout !== "horizontal")
              return;
            this.activeIndex = Math.max(this.activeIndex - 1, 0), this.spoofMouse();
            break;
          }
        }
    }
  }, {
    key: "setIndex",
    value: function(r) {
      this.activeIndex = r;
    }
  }, {
    key: "spoofMouse",
    value: function() {
      var r, n;
      if (this.layout === "horizontal" && this.coordinateList.length !== 0) {
        var i = this.container.getBoundingClientRect(), a = i.x, o = i.y, u = i.height, s = this.coordinateList[this.activeIndex].coordinate, c = ((r = window) === null || r === void 0 ? void 0 : r.scrollX) || 0, f = ((n = window) === null || n === void 0 ? void 0 : n.scrollY) || 0, l = a + s + c, d = o + this.offset.top + u / 2 + f;
        this.mouseHandlerCallback({
          pageX: l,
          pageY: d
        });
      }
    }
  }]);
}();
function tq(e, t, r) {
  if (r === "number" && t === !0 && Array.isArray(e)) {
    var n = e == null ? void 0 : e[0], i = e == null ? void 0 : e[1];
    if (n && i && K(n) && K(i))
      return !0;
  }
  return !1;
}
function rq(e, t, r, n) {
  var i = n / 2;
  return {
    stroke: "none",
    fill: "#ccc",
    x: e === "horizontal" ? t.x - i : r.left + 0.5,
    y: e === "horizontal" ? r.top + 0.5 : t.y - i,
    width: e === "horizontal" ? n : r.width - 1,
    height: e === "horizontal" ? r.height - 1 : n
  };
}
function dO(e) {
  var t = e.cx, r = e.cy, n = e.radius, i = e.startAngle, a = e.endAngle, o = Fe(t, r, n, i), u = Fe(t, r, n, a);
  return {
    points: [o, u],
    cx: t,
    cy: r,
    radius: n,
    startAngle: i,
    endAngle: a
  };
}
function nq(e, t, r) {
  var n, i, a, o;
  if (e === "horizontal")
    n = t.x, a = n, i = r.top, o = r.top + r.height;
  else if (e === "vertical")
    i = t.y, o = i, n = r.left, a = r.left + r.width;
  else if (t.cx != null && t.cy != null)
    if (e === "centric") {
      var u = t.cx, s = t.cy, c = t.innerRadius, f = t.outerRadius, l = t.angle, d = Fe(u, s, c, l), p = Fe(u, s, f, l);
      n = d.x, i = d.y, a = p.x, o = p.y;
    } else
      return dO(t);
  return [{
    x: n,
    y: i
  }, {
    x: a,
    y: o
  }];
}
function hi(e) {
  "@babel/helpers - typeof";
  return hi = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, hi(e);
}
function Kb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function zi(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Kb(Object(r), !0).forEach(function(n) {
      iq(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Kb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function iq(e, t, r) {
  return t = aq(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function aq(e) {
  var t = oq(e, "string");
  return hi(t) == "symbol" ? t : t + "";
}
function oq(e, t) {
  if (hi(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (hi(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
function uq(e) {
  var t, r, n = e.element, i = e.tooltipEventType, a = e.isActive, o = e.activeCoordinate, u = e.activePayload, s = e.offset, c = e.activeTooltipIndex, f = e.tooltipAxisBandSize, l = e.layout, d = e.chartName, p = (t = n.props.cursor) !== null && t !== void 0 ? t : (r = n.type.defaultProps) === null || r === void 0 ? void 0 : r.cursor;
  if (!n || !p || !a || !o || d !== "ScatterChart" && i !== "axis")
    return null;
  var y, v = pf;
  if (d === "ScatterChart")
    y = o, v = iR;
  else if (d === "BarChart")
    y = rq(l, o, s, f), v = Gd;
  else if (l === "radial") {
    var h = dO(o), g = h.cx, w = h.cy, b = h.radius, O = h.startAngle, m = h.endAngle;
    y = {
      cx: g,
      cy: w,
      startAngle: O,
      endAngle: m,
      innerRadius: b,
      outerRadius: b
    }, v = mw;
  } else
    y = {
      points: nq(l, o, s)
    }, v = pf;
  var x = zi(zi(zi(zi({
    stroke: "#ccc",
    pointerEvents: "none"
  }, s), y), fe(p, !1)), {}, {
    payload: u,
    payloadIndex: c,
    className: pe("recharts-tooltip-cursor", p.className)
  });
  return /* @__PURE__ */ xt(p) ? /* @__PURE__ */ De(p, x) : /* @__PURE__ */ a0(v, x);
}
var sq = ["item"], cq = ["children", "className", "width", "height", "style", "compact", "title", "desc"];
function an(e) {
  "@babel/helpers - typeof";
  return an = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(t) {
    return typeof t;
  } : function(t) {
    return t && typeof Symbol == "function" && t.constructor === Symbol && t !== Symbol.prototype ? "symbol" : typeof t;
  }, an(e);
}
function Cr() {
  return Cr = Object.assign ? Object.assign.bind() : function(e) {
    for (var t = 1; t < arguments.length; t++) {
      var r = arguments[t];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (e[n] = r[n]);
    }
    return e;
  }, Cr.apply(this, arguments);
}
function Vb(e, t) {
  return dq(e) || fq(e, t) || hO(e, t) || lq();
}
function lq() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function fq(e, t) {
  var r = e == null ? null : typeof Symbol < "u" && e[Symbol.iterator] || e["@@iterator"];
  if (r != null) {
    var n, i, a, o, u = [], s = !0, c = !1;
    try {
      if (a = (r = r.call(e)).next, t !== 0) for (; !(s = (n = a.call(r)).done) && (u.push(n.value), u.length !== t); s = !0) ;
    } catch (f) {
      c = !0, i = f;
    } finally {
      try {
        if (!s && r.return != null && (o = r.return(), Object(o) !== o)) return;
      } finally {
        if (c) throw i;
      }
    }
    return u;
  }
}
function dq(e) {
  if (Array.isArray(e)) return e;
}
function Yb(e, t) {
  if (e == null) return {};
  var r = pq(e, t), n, i;
  if (Object.getOwnPropertySymbols) {
    var a = Object.getOwnPropertySymbols(e);
    for (i = 0; i < a.length; i++)
      n = a[i], !(t.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(e, n) && (r[n] = e[n]);
  }
  return r;
}
function pq(e, t) {
  if (e == null) return {};
  var r = {};
  for (var n in e)
    if (Object.prototype.hasOwnProperty.call(e, n)) {
      if (t.indexOf(n) >= 0) continue;
      r[n] = e[n];
    }
  return r;
}
function hq(e, t) {
  if (!(e instanceof t))
    throw new TypeError("Cannot call a class as a function");
}
function vq(e, t) {
  for (var r = 0; r < t.length; r++) {
    var n = t[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(e, vO(n.key), n);
  }
}
function yq(e, t, r) {
  return vq(e.prototype, t), Object.defineProperty(e, "prototype", { writable: !1 }), e;
}
function mq(e, t, r) {
  return t = Fa(t), gq(e, pO() ? Reflect.construct(t, r, Fa(e).constructor) : t.apply(e, r));
}
function gq(e, t) {
  if (t && (an(t) === "object" || typeof t == "function"))
    return t;
  if (t !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return bq(e);
}
function bq(e) {
  if (e === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return e;
}
function pO() {
  try {
    var e = !Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    }));
  } catch {
  }
  return (pO = function() {
    return !!e;
  })();
}
function Fa(e) {
  return Fa = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Fa(e);
}
function xq(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Super expression must either be null or a function");
  e.prototype = Object.create(t && t.prototype, { constructor: { value: e, writable: !0, configurable: !0 } }), Object.defineProperty(e, "prototype", { writable: !1 }), t && Uf(e, t);
}
function Uf(e, t) {
  return Uf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Uf(e, t);
}
function on(e) {
  return _q(e) || Oq(e) || hO(e) || wq();
}
function wq() {
  throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function hO(e, t) {
  if (e) {
    if (typeof e == "string") return Wf(e, t);
    var r = Object.prototype.toString.call(e).slice(8, -1);
    if (r === "Object" && e.constructor && (r = e.constructor.name), r === "Map" || r === "Set") return Array.from(e);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)) return Wf(e, t);
  }
}
function Oq(e) {
  if (typeof Symbol < "u" && e[Symbol.iterator] != null || e["@@iterator"] != null) return Array.from(e);
}
function _q(e) {
  if (Array.isArray(e)) return Wf(e);
}
function Wf(e, t) {
  (t == null || t > e.length) && (t = e.length);
  for (var r = 0, n = new Array(t); r < t; r++) n[r] = e[r];
  return n;
}
function Xb(e, t) {
  var r = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(e);
    t && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(e, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function N(e) {
  for (var t = 1; t < arguments.length; t++) {
    var r = arguments[t] != null ? arguments[t] : {};
    t % 2 ? Xb(Object(r), !0).forEach(function(n) {
      ie(e, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Xb(Object(r)).forEach(function(n) {
      Object.defineProperty(e, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return e;
}
function ie(e, t, r) {
  return t = vO(t), t in e ? Object.defineProperty(e, t, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : e[t] = r, e;
}
function vO(e) {
  var t = Sq(e, "string");
  return an(t) == "symbol" ? t : t + "";
}
function Sq(e, t) {
  if (an(e) != "object" || !e) return e;
  var r = e[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(e, t || "default");
    if (an(n) != "object") return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (t === "string" ? String : Number)(e);
}
var Pq = {
  xAxis: ["bottom", "top"],
  yAxis: ["left", "right"]
}, Aq = {
  width: "100%",
  height: "100%"
}, yO = {
  x: 0,
  y: 0
};
function Ui(e) {
  return e;
}
var Eq = function(t, r) {
  return r === "horizontal" ? t.x : r === "vertical" ? t.y : r === "centric" ? t.angle : t.radius;
}, Tq = function(t, r, n, i) {
  var a = r.find(function(f) {
    return f && f.index === n;
  });
  if (a) {
    if (t === "horizontal")
      return {
        x: a.coordinate,
        y: i.y
      };
    if (t === "vertical")
      return {
        x: i.x,
        y: a.coordinate
      };
    if (t === "centric") {
      var o = a.coordinate, u = i.radius;
      return N(N(N({}, i), Fe(i.cx, i.cy, u, o)), {}, {
        angle: o,
        radius: u
      });
    }
    var s = a.coordinate, c = i.angle;
    return N(N(N({}, i), Fe(i.cx, i.cy, s, c)), {}, {
      angle: c,
      radius: s
    });
  }
  return yO;
}, Oo = function(t, r) {
  var n = r.graphicalItems, i = r.dataStartIndex, a = r.dataEndIndex, o = (n ?? []).reduce(function(u, s) {
    var c = s.props.data;
    return c && c.length ? [].concat(on(u), on(c)) : u;
  }, []);
  return o.length > 0 ? o : t && t.length && K(i) && K(a) ? t.slice(i, a + 1) : [];
};
function mO(e) {
  return e === "number" ? [0, "auto"] : void 0;
}
var Gf = function(t, r, n, i) {
  var a = t.graphicalItems, o = t.tooltipAxis, u = Oo(r, t);
  return n < 0 || !a || !a.length || n >= u.length ? null : a.reduce(function(s, c) {
    var f, l = (f = c.props.data) !== null && f !== void 0 ? f : r;
    l && t.dataStartIndex + t.dataEndIndex !== 0 && // https://github.com/recharts/recharts/issues/4717
    // The data is sliced only when the active index is within the start/end index range.
    t.dataEndIndex - t.dataStartIndex >= n && (l = l.slice(t.dataStartIndex, t.dataEndIndex + 1));
    var d;
    if (o.dataKey && !o.allowDuplicatedCategory) {
      var p = l === void 0 ? u : l;
      d = Wi(p, o.dataKey, i);
    } else
      d = l && l[n] || u[n];
    return d ? [].concat(on(s), [hw(c, d)]) : s;
  }, []);
}, Zb = function(t, r, n, i) {
  var a = i || {
    x: t.chartX,
    y: t.chartY
  }, o = Eq(a, n), u = t.orderedTooltipTicks, s = t.tooltipAxis, c = t.tooltipTicks, f = NC(o, u, c, s);
  if (f >= 0 && c) {
    var l = c[f] && c[f].value, d = Gf(t, r, f, l), p = Tq(n, u, f, a);
    return {
      activeTooltipIndex: f,
      activeLabel: l,
      activePayload: d,
      activeCoordinate: p
    };
  }
  return null;
}, jq = function(t, r) {
  var n = r.axes, i = r.graphicalItems, a = r.axisType, o = r.axisIdKey, u = r.stackGroups, s = r.dataStartIndex, c = r.dataEndIndex, f = t.layout, l = t.children, d = t.stackOffset, p = fw(f, a);
  return n.reduce(function(y, v) {
    var h, g = v.type.defaultProps !== void 0 ? N(N({}, v.type.defaultProps), v.props) : v.props, w = g.type, b = g.dataKey, O = g.allowDataOverflow, m = g.allowDuplicatedCategory, x = g.scale, _ = g.ticks, P = g.includeHidden, E = g[o];
    if (y[E])
      return y;
    var I = Oo(t.data, {
      graphicalItems: i.filter(function(B) {
        var G, V = o in B.props ? B.props[o] : (G = B.type.defaultProps) === null || G === void 0 ? void 0 : G[o];
        return V === E;
      }),
      dataStartIndex: s,
      dataEndIndex: c
    }), S = I.length, j, M, R;
    tq(g.domain, O, w) && (j = sf(g.domain, null, O), p && (w === "number" || x !== "auto") && (R = $n(I, b, "category")));
    var k = mO(w);
    if (!j || j.length === 0) {
      var q, L = (q = g.domain) !== null && q !== void 0 ? q : k;
      if (b) {
        if (j = $n(I, b, w), w === "category" && p) {
          var U = R_(j);
          m && U ? (M = j, j = Ea(0, S)) : m || (j = tg(L, j, v).reduce(function(B, G) {
            return B.indexOf(G) >= 0 ? B : [].concat(on(B), [G]);
          }, []));
        } else if (w === "category")
          m ? j = j.filter(function(B) {
            return B !== "" && !ce(B);
          }) : j = tg(L, j, v).reduce(function(B, G) {
            return B.indexOf(G) >= 0 || G === "" || ce(G) ? B : [].concat(on(B), [G]);
          }, []);
        else if (w === "number") {
          var z = FC(I, i.filter(function(B) {
            var G, V, te = o in B.props ? B.props[o] : (G = B.type.defaultProps) === null || G === void 0 ? void 0 : G[o], re = "hide" in B.props ? B.props.hide : (V = B.type.defaultProps) === null || V === void 0 ? void 0 : V.hide;
            return te === E && (P || !re);
          }), b, a, f);
          z && (j = z);
        }
        p && (w === "number" || x !== "auto") && (R = $n(I, b, "category"));
      } else p ? j = Ea(0, S) : u && u[E] && u[E].hasStack && w === "number" ? j = d === "expand" ? [0, 1] : pw(u[E].stackGroups, s, c) : j = lw(I, i.filter(function(B) {
        var G = o in B.props ? B.props[o] : B.type.defaultProps[o], V = "hide" in B.props ? B.props.hide : B.type.defaultProps.hide;
        return G === E && (P || !V);
      }), w, f, !0);
      if (w === "number")
        j = zf(l, j, E, a, _), L && (j = sf(L, j, O));
      else if (w === "category" && L) {
        var $ = L, D = j.every(function(B) {
          return $.indexOf(B) >= 0;
        });
        D && (j = $);
      }
    }
    return N(N({}, y), {}, ie({}, E, N(N({}, g), {}, {
      axisType: a,
      domain: j,
      categoricalDomain: R,
      duplicateDomain: M,
      originalDomain: (h = g.domain) !== null && h !== void 0 ? h : k,
      isCategorical: p,
      layout: f
    })));
  }, {});
}, Cq = function(t, r) {
  var n = r.graphicalItems, i = r.Axis, a = r.axisType, o = r.axisIdKey, u = r.stackGroups, s = r.dataStartIndex, c = r.dataEndIndex, f = t.layout, l = t.children, d = Oo(t.data, {
    graphicalItems: n,
    dataStartIndex: s,
    dataEndIndex: c
  }), p = d.length, y = fw(f, a), v = -1;
  return n.reduce(function(h, g) {
    var w = g.type.defaultProps !== void 0 ? N(N({}, g.type.defaultProps), g.props) : g.props, b = w[o], O = mO("number");
    if (!h[b]) {
      v++;
      var m;
      return y ? m = Ea(0, p) : u && u[b] && u[b].hasStack ? (m = pw(u[b].stackGroups, s, c), m = zf(l, m, b, a)) : (m = sf(O, lw(d, n.filter(function(x) {
        var _, P, E = o in x.props ? x.props[o] : (_ = x.type.defaultProps) === null || _ === void 0 ? void 0 : _[o], I = "hide" in x.props ? x.props.hide : (P = x.type.defaultProps) === null || P === void 0 ? void 0 : P.hide;
        return E === b && !I;
      }), "number", f), i.defaultProps.allowDataOverflow), m = zf(l, m, b, a)), N(N({}, h), {}, ie({}, b, N(N({
        axisType: a
      }, i.defaultProps), {}, {
        hide: !0,
        orientation: at(Pq, "".concat(a, ".").concat(v % 2), null),
        domain: m,
        originalDomain: O,
        isCategorical: y,
        layout: f
        // specify scale when no Axis
        // scale: isCategorical ? 'band' : 'linear',
      })));
    }
    return h;
  }, {});
}, Mq = function(t, r) {
  var n = r.axisType, i = n === void 0 ? "xAxis" : n, a = r.AxisComp, o = r.graphicalItems, u = r.stackGroups, s = r.dataStartIndex, c = r.dataEndIndex, f = t.children, l = "".concat(i, "Id"), d = ot(f, a), p = {};
  return d.length ? p = jq(t, {
    axes: d,
    graphicalItems: o,
    axisType: i,
    axisIdKey: l,
    stackGroups: u,
    dataStartIndex: s,
    dataEndIndex: c
  }) : o && o.length && (p = Cq(t, {
    Axis: a,
    graphicalItems: o,
    axisType: i,
    axisIdKey: l,
    stackGroups: u,
    dataStartIndex: s,
    dataEndIndex: c
  })), p;
}, Iq = function(t) {
  var r = Wt(t), n = Mt(r, !1, !0);
  return {
    tooltipTicks: n,
    orderedTooltipTicks: md(n, function(i) {
      return i.coordinate;
    }),
    tooltipAxis: r,
    tooltipAxisBandSize: ma(r, n)
  };
}, Jb = function(t) {
  var r = t.children, n = t.defaultShowTooltip, i = Qe(r, Kr), a = 0, o = 0;
  return t.data && t.data.length !== 0 && (o = t.data.length - 1), i && i.props && (i.props.startIndex >= 0 && (a = i.props.startIndex), i.props.endIndex >= 0 && (o = i.props.endIndex)), {
    chartX: 0,
    chartY: 0,
    dataStartIndex: a,
    dataEndIndex: o,
    activeTooltipIndex: -1,
    isTooltipActive: !!n
  };
}, $q = function(t) {
  return !t || !t.length ? !1 : t.some(function(r) {
    var n = Ht(r && r.type);
    return n && n.indexOf("Bar") >= 0;
  });
}, Qb = function(t) {
  return t === "horizontal" ? {
    numericAxisName: "yAxis",
    cateAxisName: "xAxis"
  } : t === "vertical" ? {
    numericAxisName: "xAxis",
    cateAxisName: "yAxis"
  } : t === "centric" ? {
    numericAxisName: "radiusAxis",
    cateAxisName: "angleAxis"
  } : {
    numericAxisName: "angleAxis",
    cateAxisName: "radiusAxis"
  };
}, Rq = function(t, r) {
  var n = t.props, i = t.graphicalItems, a = t.xAxisMap, o = a === void 0 ? {} : a, u = t.yAxisMap, s = u === void 0 ? {} : u, c = n.width, f = n.height, l = n.children, d = n.margin || {}, p = Qe(l, Kr), y = Qe(l, Ir), v = Object.keys(s).reduce(function(m, x) {
    var _ = s[x], P = _.orientation;
    return !_.mirror && !_.hide ? N(N({}, m), {}, ie({}, P, m[P] + _.width)) : m;
  }, {
    left: d.left || 0,
    right: d.right || 0
  }), h = Object.keys(o).reduce(function(m, x) {
    var _ = o[x], P = _.orientation;
    return !_.mirror && !_.hide ? N(N({}, m), {}, ie({}, P, at(m, "".concat(P)) + _.height)) : m;
  }, {
    top: d.top || 0,
    bottom: d.bottom || 0
  }), g = N(N({}, h), v), w = g.bottom;
  p && (g.bottom += p.props.height || Kr.defaultProps.height), y && r && (g = LC(g, i, n, r));
  var b = c - g.left - g.right, O = f - g.top - g.bottom;
  return N(N({
    brushBottom: w
  }, g), {}, {
    // never return negative values for height and width
    width: Math.max(b, 0),
    height: Math.max(O, 0)
  });
}, kq = function(t, r) {
  if (r === "xAxis")
    return t[r].width;
  if (r === "yAxis")
    return t[r].height;
}, Nq = function(t) {
  var r = t.chartName, n = t.GraphicalChild, i = t.defaultTooltipEventType, a = i === void 0 ? "axis" : i, o = t.validateTooltipEventTypes, u = o === void 0 ? ["axis"] : o, s = t.axisComponents, c = t.legendContent, f = t.formatAxisMap, l = t.defaultProps, d = function(g, w) {
    var b = w.graphicalItems, O = w.stackGroups, m = w.offset, x = w.updateId, _ = w.dataStartIndex, P = w.dataEndIndex, E = g.barSize, I = g.layout, S = g.barGap, j = g.barCategoryGap, M = g.maxBarSize, R = Qb(I), k = R.numericAxisName, q = R.cateAxisName, L = $q(b), U = [];
    return b.forEach(function(z, $) {
      var D = Oo(g.data, {
        graphicalItems: [z],
        dataStartIndex: _,
        dataEndIndex: P
      }), B = z.type.defaultProps !== void 0 ? N(N({}, z.type.defaultProps), z.props) : z.props, G = B.dataKey, V = B.maxBarSize, te = B["".concat(k, "Id")], re = B["".concat(q, "Id")], ae = {}, ne = s.reduce(function(A, X) {
        var J, le, Me = w["".concat(X.axisType, "Map")], _e = B["".concat(X.axisType, "Id")];
        Me && Me[_e] || X.axisType === "zAxis" || (process.env.NODE_ENV !== "production" ? Ye(!1, "Specifying a(n) ".concat(X.axisType, "Id requires a corresponding ").concat(
          X.axisType,
          "Id on the targeted graphical component "
        ).concat((J = z == null || (le = z.type) === null || le === void 0 ? void 0 : le.displayName) !== null && J !== void 0 ? J : "")) : Ye());
        var oe = Me[_e];
        return N(N({}, A), {}, ie(ie({}, X.axisType, oe), "".concat(X.axisType, "Ticks"), Mt(oe)));
      }, ae), F = ne[q], H = ne["".concat(q, "Ticks")], ee = O && O[te] && O[te].hasStack && QC(z, O[te].stackGroups), C = Ht(z.type).indexOf("Bar") >= 0, se = ma(F, H), W = [], he = L && DC({
        barSize: E,
        stackGroups: O,
        totalSize: kq(ne, q)
      });
      if (C) {
        var Oe, Ce, ct = ce(V) ? M : V, Pt = (Oe = (Ce = ma(F, H, !0)) !== null && Ce !== void 0 ? Ce : ct) !== null && Oe !== void 0 ? Oe : 0;
        W = qC({
          barGap: S,
          barCategoryGap: j,
          bandSize: Pt !== se ? Pt : se,
          sizeList: he[re],
          maxBarSize: ct
        }), Pt !== se && (W = W.map(function(A) {
          return N(N({}, A), {}, {
            position: N(N({}, A.position), {}, {
              offset: A.position.offset - Pt / 2
            })
          });
        }));
      }
      var rr = z && z.type && z.type.getComposedData;
      rr && U.push({
        props: N(N({}, rr(N(N({}, ne), {}, {
          displayedData: D,
          props: g,
          dataKey: G,
          item: z,
          bandSize: se,
          barPosition: W,
          offset: m,
          stackedData: ee,
          layout: I,
          dataStartIndex: _,
          dataEndIndex: P
        }))), {}, ie(ie(ie({
          key: z.key || "item-".concat($)
        }, k, ne[k]), q, ne[q]), "animationId", x)),
        childIndex: H_(z, g.children),
        item: z
      });
    }), U;
  }, p = function(g, w) {
    var b = g.props, O = g.dataStartIndex, m = g.dataEndIndex, x = g.updateId;
    if (!mh({
      props: b
    }))
      return null;
    var _ = b.children, P = b.layout, E = b.stackOffset, I = b.data, S = b.reverseStackOrder, j = Qb(P), M = j.numericAxisName, R = j.cateAxisName, k = ot(_, n), q = XC(I, k, "".concat(M, "Id"), "".concat(R, "Id"), E, S), L = s.reduce(function(B, G) {
      var V = "".concat(G.axisType, "Map");
      return N(N({}, B), {}, ie({}, V, Mq(b, N(N({}, G), {}, {
        graphicalItems: k,
        stackGroups: G.axisType === M && q,
        dataStartIndex: O,
        dataEndIndex: m
      }))));
    }, {}), U = Rq(N(N({}, L), {}, {
      props: b,
      graphicalItems: k
    }), w == null ? void 0 : w.legendBBox);
    Object.keys(L).forEach(function(B) {
      L[B] = f(b, L[B], U, B.replace("Map", ""), r);
    });
    var z = L["".concat(R, "Map")], $ = Iq(z), D = d(b, N(N({}, L), {}, {
      dataStartIndex: O,
      dataEndIndex: m,
      updateId: x,
      graphicalItems: k,
      stackGroups: q,
      offset: U
    }));
    return N(N({
      formattedGraphicalItems: D,
      graphicalItems: k,
      offset: U,
      stackGroups: q
    }, $), L);
  }, y = /* @__PURE__ */ function(h) {
    function g(w) {
      var b, O, m;
      return hq(this, g), m = mq(this, g, [w]), ie(m, "eventEmitterSymbol", Symbol("rechartsEventEmitter")), ie(m, "accessibilityManager", new eq()), ie(m, "handleLegendBBoxUpdate", function(x) {
        if (x) {
          var _ = m.state, P = _.dataStartIndex, E = _.dataEndIndex, I = _.updateId;
          m.setState(N({
            legendBBox: x
          }, p({
            props: m.props,
            dataStartIndex: P,
            dataEndIndex: E,
            updateId: I
          }, N(N({}, m.state), {}, {
            legendBBox: x
          }))));
        }
      }), ie(m, "handleReceiveSyncEvent", function(x, _, P) {
        if (m.props.syncId === x) {
          if (P === m.eventEmitterSymbol && typeof m.props.syncMethod != "function")
            return;
          m.applySyncEvent(_);
        }
      }), ie(m, "handleBrushChange", function(x) {
        var _ = x.startIndex, P = x.endIndex;
        if (_ !== m.state.dataStartIndex || P !== m.state.dataEndIndex) {
          var E = m.state.updateId;
          m.setState(function() {
            return N({
              dataStartIndex: _,
              dataEndIndex: P
            }, p({
              props: m.props,
              dataStartIndex: _,
              dataEndIndex: P,
              updateId: E
            }, m.state));
          }), m.triggerSyncEvent({
            dataStartIndex: _,
            dataEndIndex: P
          });
        }
      }), ie(m, "handleMouseEnter", function(x) {
        var _ = m.getMouseInfo(x);
        if (_) {
          var P = N(N({}, _), {}, {
            isTooltipActive: !0
          });
          m.setState(P), m.triggerSyncEvent(P);
          var E = m.props.onMouseEnter;
          ue(E) && E(P, x);
        }
      }), ie(m, "triggeredAfterMouseMove", function(x) {
        var _ = m.getMouseInfo(x), P = _ ? N(N({}, _), {}, {
          isTooltipActive: !0
        }) : {
          isTooltipActive: !1
        };
        m.setState(P), m.triggerSyncEvent(P);
        var E = m.props.onMouseMove;
        ue(E) && E(P, x);
      }), ie(m, "handleItemMouseEnter", function(x) {
        m.setState(function() {
          return {
            isTooltipActive: !0,
            activeItem: x,
            activePayload: x.tooltipPayload,
            activeCoordinate: x.tooltipPosition || {
              x: x.cx,
              y: x.cy
            }
          };
        });
      }), ie(m, "handleItemMouseLeave", function() {
        m.setState(function() {
          return {
            isTooltipActive: !1
          };
        });
      }), ie(m, "handleMouseMove", function(x) {
        x.persist(), m.throttleTriggeredAfterMouseMove(x);
      }), ie(m, "handleMouseLeave", function(x) {
        m.throttleTriggeredAfterMouseMove.cancel();
        var _ = {
          isTooltipActive: !1
        };
        m.setState(_), m.triggerSyncEvent(_);
        var P = m.props.onMouseLeave;
        ue(P) && P(_, x);
      }), ie(m, "handleOuterEvent", function(x) {
        var _ = G_(x), P = at(m.props, "".concat(_));
        if (_ && ue(P)) {
          var E, I;
          /.*touch.*/i.test(_) ? I = m.getMouseInfo(x.changedTouches[0]) : I = m.getMouseInfo(x), P((E = I) !== null && E !== void 0 ? E : {}, x);
        }
      }), ie(m, "handleClick", function(x) {
        var _ = m.getMouseInfo(x);
        if (_) {
          var P = N(N({}, _), {}, {
            isTooltipActive: !0
          });
          m.setState(P), m.triggerSyncEvent(P);
          var E = m.props.onClick;
          ue(E) && E(P, x);
        }
      }), ie(m, "handleMouseDown", function(x) {
        var _ = m.props.onMouseDown;
        if (ue(_)) {
          var P = m.getMouseInfo(x);
          _(P, x);
        }
      }), ie(m, "handleMouseUp", function(x) {
        var _ = m.props.onMouseUp;
        if (ue(_)) {
          var P = m.getMouseInfo(x);
          _(P, x);
        }
      }), ie(m, "handleTouchMove", function(x) {
        x.changedTouches != null && x.changedTouches.length > 0 && m.throttleTriggeredAfterMouseMove(x.changedTouches[0]);
      }), ie(m, "handleTouchStart", function(x) {
        x.changedTouches != null && x.changedTouches.length > 0 && m.handleMouseDown(x.changedTouches[0]);
      }), ie(m, "handleTouchEnd", function(x) {
        x.changedTouches != null && x.changedTouches.length > 0 && m.handleMouseUp(x.changedTouches[0]);
      }), ie(m, "handleDoubleClick", function(x) {
        var _ = m.props.onDoubleClick;
        if (ue(_)) {
          var P = m.getMouseInfo(x);
          _(P, x);
        }
      }), ie(m, "handleContextMenu", function(x) {
        var _ = m.props.onContextMenu;
        if (ue(_)) {
          var P = m.getMouseInfo(x);
          _(P, x);
        }
      }), ie(m, "triggerSyncEvent", function(x) {
        m.props.syncId !== void 0 && _l.emit(Sl, m.props.syncId, x, m.eventEmitterSymbol);
      }), ie(m, "applySyncEvent", function(x) {
        var _ = m.props, P = _.layout, E = _.syncMethod, I = m.state.updateId, S = x.dataStartIndex, j = x.dataEndIndex;
        if (x.dataStartIndex !== void 0 || x.dataEndIndex !== void 0)
          m.setState(N({
            dataStartIndex: S,
            dataEndIndex: j
          }, p({
            props: m.props,
            dataStartIndex: S,
            dataEndIndex: j,
            updateId: I
          }, m.state)));
        else if (x.activeTooltipIndex !== void 0) {
          var M = x.chartX, R = x.chartY, k = x.activeTooltipIndex, q = m.state, L = q.offset, U = q.tooltipTicks;
          if (!L)
            return;
          if (typeof E == "function")
            k = E(U, x);
          else if (E === "value") {
            k = -1;
            for (var z = 0; z < U.length; z++)
              if (U[z].value === x.activeLabel) {
                k = z;
                break;
              }
          }
          var $ = N(N({}, L), {}, {
            x: L.left,
            y: L.top
          }), D = Math.min(M, $.x + $.width), B = Math.min(R, $.y + $.height), G = U[k] && U[k].value, V = Gf(m.state, m.props.data, k), te = U[k] ? {
            x: P === "horizontal" ? U[k].coordinate : D,
            y: P === "horizontal" ? B : U[k].coordinate
          } : yO;
          m.setState(N(N({}, x), {}, {
            activeLabel: G,
            activeCoordinate: te,
            activePayload: V,
            activeTooltipIndex: k
          }));
        } else
          m.setState(x);
      }), ie(m, "renderCursor", function(x) {
        var _, P = m.state, E = P.isTooltipActive, I = P.activeCoordinate, S = P.activePayload, j = P.offset, M = P.activeTooltipIndex, R = P.tooltipAxisBandSize, k = m.getTooltipEventType(), q = (_ = x.props.active) !== null && _ !== void 0 ? _ : E, L = m.props.layout, U = x.key || "_recharts-cursor";
        return /* @__PURE__ */ T.createElement(uq, {
          key: U,
          activeCoordinate: I,
          activePayload: S,
          activeTooltipIndex: M,
          chartName: r,
          element: x,
          isActive: q,
          layout: L,
          offset: j,
          tooltipAxisBandSize: R,
          tooltipEventType: k
        });
      }), ie(m, "renderPolarAxis", function(x, _, P) {
        var E = at(x, "type.axisType"), I = at(m.state, "".concat(E, "Map")), S = x.type.defaultProps, j = S !== void 0 ? N(N({}, S), x.props) : x.props, M = I && I[j["".concat(E, "Id")]];
        return /* @__PURE__ */ De(x, N(N({}, M), {}, {
          className: pe(E, M.className),
          key: x.key || "".concat(_, "-").concat(P),
          ticks: Mt(M, !0)
        }));
      }), ie(m, "renderPolarGrid", function(x) {
        var _ = x.props, P = _.radialLines, E = _.polarAngles, I = _.polarRadius, S = m.state, j = S.radiusAxisMap, M = S.angleAxisMap, R = Wt(j), k = Wt(M), q = k.cx, L = k.cy, U = k.innerRadius, z = k.outerRadius;
        return /* @__PURE__ */ De(x, {
          polarAngles: Array.isArray(E) ? E : Mt(k, !0).map(function($) {
            return $.coordinate;
          }),
          polarRadius: Array.isArray(I) ? I : Mt(R, !0).map(function($) {
            return $.coordinate;
          }),
          cx: q,
          cy: L,
          innerRadius: U,
          outerRadius: z,
          key: x.key || "polar-grid",
          radialLines: P
        });
      }), ie(m, "renderLegend", function() {
        var x = m.state.formattedGraphicalItems, _ = m.props, P = _.children, E = _.width, I = _.height, S = m.props.margin || {}, j = E - (S.left || 0) - (S.right || 0), M = sw({
          children: P,
          formattedGraphicalItems: x,
          legendWidth: j,
          legendContent: c
        });
        if (!M)
          return null;
        var R = M.item, k = Yb(M, sq);
        return /* @__PURE__ */ De(R, N(N({}, k), {}, {
          chartWidth: E,
          chartHeight: I,
          margin: S,
          onBBoxUpdate: m.handleLegendBBoxUpdate
        }));
      }), ie(m, "renderTooltip", function() {
        var x, _ = m.props, P = _.children, E = _.accessibilityLayer, I = Qe(P, gt);
        if (!I)
          return null;
        var S = m.state, j = S.isTooltipActive, M = S.activeCoordinate, R = S.activePayload, k = S.activeLabel, q = S.offset, L = (x = I.props.active) !== null && x !== void 0 ? x : j;
        return /* @__PURE__ */ De(I, {
          viewBox: N(N({}, q), {}, {
            x: q.left,
            y: q.top
          }),
          active: L,
          label: k,
          payload: L ? R : [],
          coordinate: M,
          accessibilityLayer: E
        });
      }), ie(m, "renderBrush", function(x) {
        var _ = m.props, P = _.margin, E = _.data, I = m.state, S = I.offset, j = I.dataStartIndex, M = I.dataEndIndex, R = I.updateId;
        return /* @__PURE__ */ De(x, {
          key: x.key || "_recharts-brush",
          onChange: Ni(m.handleBrushChange, x.props.onChange),
          data: E,
          x: K(x.props.x) ? x.props.x : S.left,
          y: K(x.props.y) ? x.props.y : S.top + S.height + S.brushBottom - (P.bottom || 0),
          width: K(x.props.width) ? x.props.width : S.width,
          startIndex: j,
          endIndex: M,
          updateId: "brush-".concat(R)
        });
      }), ie(m, "renderReferenceElement", function(x, _, P) {
        if (!x)
          return null;
        var E = m, I = E.clipPathId, S = m.state, j = S.xAxisMap, M = S.yAxisMap, R = S.offset, k = x.type.defaultProps || {}, q = x.props, L = q.xAxisId, U = L === void 0 ? k.xAxisId : L, z = q.yAxisId, $ = z === void 0 ? k.yAxisId : z;
        return /* @__PURE__ */ De(x, {
          key: x.key || "".concat(_, "-").concat(P),
          xAxis: j[U],
          yAxis: M[$],
          viewBox: {
            x: R.left,
            y: R.top,
            width: R.width,
            height: R.height
          },
          clipPathId: I
        });
      }), ie(m, "renderActivePoints", function(x) {
        var _ = x.item, P = x.activePoint, E = x.basePoint, I = x.childIndex, S = x.isRange, j = [], M = _.props.key, R = _.item.type.defaultProps !== void 0 ? N(N({}, _.item.type.defaultProps), _.item.props) : _.item.props, k = R.activeDot, q = R.dataKey, L = N(N({
          index: I,
          dataKey: q,
          cx: P.x,
          cy: P.y,
          r: 4,
          fill: Ud(_.item),
          strokeWidth: 2,
          stroke: "#fff",
          payload: P.payload,
          value: P.value
        }, fe(k, !1)), Gi(k));
        return j.push(g.renderActiveDot(k, L, "".concat(M, "-activePoint-").concat(I))), E ? j.push(g.renderActiveDot(k, N(N({}, L), {}, {
          cx: E.x,
          cy: E.y
        }), "".concat(M, "-basePoint-").concat(I))) : S && j.push(null), j;
      }), ie(m, "renderGraphicChild", function(x, _, P) {
        var E = m.filterFormatItem(x, _, P);
        if (!E)
          return null;
        var I = m.getTooltipEventType(), S = m.state, j = S.isTooltipActive, M = S.tooltipAxis, R = S.activeTooltipIndex, k = S.activeLabel, q = m.props.children, L = Qe(q, gt), U = E.props, z = U.points, $ = U.isRange, D = U.baseLine, B = E.item.type.defaultProps !== void 0 ? N(N({}, E.item.type.defaultProps), E.item.props) : E.item.props, G = B.activeDot, V = B.hide, te = B.activeBar, re = B.activeShape, ae = !!(!V && j && L && (G || te || re)), ne = {};
        I !== "axis" && L && L.props.trigger === "click" ? ne = {
          onClick: Ni(m.handleItemMouseEnter, x.props.onClick)
        } : I !== "axis" && (ne = {
          onMouseLeave: Ni(m.handleItemMouseLeave, x.props.onMouseLeave),
          onMouseEnter: Ni(m.handleItemMouseEnter, x.props.onMouseEnter)
        });
        var F = /* @__PURE__ */ De(x, N(N({}, E.props), ne));
        function H(X) {
          return typeof M.dataKey == "function" ? M.dataKey(X.payload) : null;
        }
        if (ae)
          if (R >= 0) {
            var ee, C;
            if (M.dataKey && !M.allowDuplicatedCategory) {
              var se = typeof M.dataKey == "function" ? H : "payload.".concat(M.dataKey.toString());
              ee = Wi(z, se, k), C = $ && D && Wi(D, se, k);
            } else
              ee = z == null ? void 0 : z[R], C = $ && D && D[R];
            if (re || te) {
              var W = x.props.activeIndex !== void 0 ? x.props.activeIndex : R;
              return [/* @__PURE__ */ De(x, N(N(N({}, E.props), ne), {}, {
                activeIndex: W
              })), null, null];
            }
            if (!ce(ee))
              return [F].concat(on(m.renderActivePoints({
                item: E,
                activePoint: ee,
                basePoint: C,
                childIndex: R,
                isRange: $
              })));
          } else {
            var he, Oe = (he = m.getItemByXY(m.state.activeCoordinate)) !== null && he !== void 0 ? he : {
              graphicalItem: F
            }, Ce = Oe.graphicalItem, ct = Ce.item, Pt = ct === void 0 ? x : ct, rr = Ce.childIndex, A = N(N(N({}, E.props), ne), {}, {
              activeIndex: rr
            });
            return [/* @__PURE__ */ De(Pt, A), null, null];
          }
        return $ ? [F, null, null] : [F, null];
      }), ie(m, "renderCustomized", function(x, _, P) {
        return /* @__PURE__ */ De(x, N(N({
          key: "recharts-customized-".concat(P)
        }, m.props), m.state));
      }), ie(m, "renderMap", {
        CartesianGrid: {
          handler: Ui,
          once: !0
        },
        ReferenceArea: {
          handler: m.renderReferenceElement
        },
        ReferenceLine: {
          handler: Ui
        },
        ReferenceDot: {
          handler: m.renderReferenceElement
        },
        XAxis: {
          handler: Ui
        },
        YAxis: {
          handler: Ui
        },
        Brush: {
          handler: m.renderBrush,
          once: !0
        },
        Bar: {
          handler: m.renderGraphicChild
        },
        Line: {
          handler: m.renderGraphicChild
        },
        Area: {
          handler: m.renderGraphicChild
        },
        Radar: {
          handler: m.renderGraphicChild
        },
        RadialBar: {
          handler: m.renderGraphicChild
        },
        Scatter: {
          handler: m.renderGraphicChild
        },
        Pie: {
          handler: m.renderGraphicChild
        },
        Funnel: {
          handler: m.renderGraphicChild
        },
        Tooltip: {
          handler: m.renderCursor,
          once: !0
        },
        PolarGrid: {
          handler: m.renderPolarGrid,
          once: !0
        },
        PolarAngleAxis: {
          handler: m.renderPolarAxis
        },
        PolarRadiusAxis: {
          handler: m.renderPolarAxis
        },
        Customized: {
          handler: m.renderCustomized
        }
      }), m.clipPathId = "".concat((b = w.id) !== null && b !== void 0 ? b : gi("recharts"), "-clip"), m.throttleTriggeredAfterMouseMove = bA(m.triggeredAfterMouseMove, (O = w.throttleDelay) !== null && O !== void 0 ? O : 1e3 / 60), m.state = {}, m;
    }
    return xq(g, h), yq(g, [{
      key: "componentDidMount",
      value: function() {
        var b, O;
        this.addListener(), this.accessibilityManager.setDetails({
          container: this.container,
          offset: {
            left: (b = this.props.margin.left) !== null && b !== void 0 ? b : 0,
            top: (O = this.props.margin.top) !== null && O !== void 0 ? O : 0
          },
          coordinateList: this.state.tooltipTicks,
          mouseHandlerCallback: this.triggeredAfterMouseMove,
          layout: this.props.layout
        }), this.displayDefaultTooltip();
      }
    }, {
      key: "displayDefaultTooltip",
      value: function() {
        var b = this.props, O = b.children, m = b.data, x = b.height, _ = b.layout, P = Qe(O, gt);
        if (P) {
          var E = P.props.defaultIndex;
          if (!(typeof E != "number" || E < 0 || E > this.state.tooltipTicks.length - 1)) {
            var I = this.state.tooltipTicks[E] && this.state.tooltipTicks[E].value, S = Gf(this.state, m, E, I), j = this.state.tooltipTicks[E].coordinate, M = (this.state.offset.top + x) / 2, R = _ === "horizontal", k = R ? {
              x: j,
              y: M
            } : {
              y: j,
              x: M
            }, q = this.state.formattedGraphicalItems.find(function(U) {
              var z = U.item;
              return z.type.name === "Scatter";
            });
            q && (k = N(N({}, k), q.props.points[E].tooltipPosition), S = q.props.points[E].tooltipPayload);
            var L = {
              activeTooltipIndex: E,
              isTooltipActive: !0,
              activeLabel: I,
              activePayload: S,
              activeCoordinate: k
            };
            this.setState(L), this.renderCursor(P), this.accessibilityManager.setIndex(E);
          }
        }
      }
    }, {
      key: "getSnapshotBeforeUpdate",
      value: function(b, O) {
        if (!this.props.accessibilityLayer)
          return null;
        if (this.state.tooltipTicks !== O.tooltipTicks && this.accessibilityManager.setDetails({
          coordinateList: this.state.tooltipTicks
        }), this.props.layout !== b.layout && this.accessibilityManager.setDetails({
          layout: this.props.layout
        }), this.props.margin !== b.margin) {
          var m, x;
          this.accessibilityManager.setDetails({
            offset: {
              left: (m = this.props.margin.left) !== null && m !== void 0 ? m : 0,
              top: (x = this.props.margin.top) !== null && x !== void 0 ? x : 0
            }
          });
        }
        return null;
      }
    }, {
      key: "componentDidUpdate",
      value: function(b) {
        jl([Qe(b.children, gt)], [Qe(this.props.children, gt)]) || this.displayDefaultTooltip();
      }
    }, {
      key: "componentWillUnmount",
      value: function() {
        this.removeListener(), this.throttleTriggeredAfterMouseMove.cancel();
      }
    }, {
      key: "getTooltipEventType",
      value: function() {
        var b = Qe(this.props.children, gt);
        if (b && typeof b.props.shared == "boolean") {
          var O = b.props.shared ? "axis" : "item";
          return u.indexOf(O) >= 0 ? O : a;
        }
        return a;
      }
      /**
       * Get the information of mouse in chart, return null when the mouse is not in the chart
       * @param  {MousePointer} event    The event object
       * @return {Object}          Mouse data
       */
    }, {
      key: "getMouseInfo",
      value: function(b) {
        if (!this.container)
          return null;
        var O = this.container, m = O.getBoundingClientRect(), x = AA(m), _ = {
          chartX: Math.round(b.pageX - x.left),
          chartY: Math.round(b.pageY - x.top)
        }, P = m.width / O.offsetWidth || 1, E = this.inRange(_.chartX, _.chartY, P);
        if (!E)
          return null;
        var I = this.state, S = I.xAxisMap, j = I.yAxisMap, M = this.getTooltipEventType();
        if (M !== "axis" && S && j) {
          var R = Wt(S).scale, k = Wt(j).scale, q = R && R.invert ? R.invert(_.chartX) : null, L = k && k.invert ? k.invert(_.chartY) : null;
          return N(N({}, _), {}, {
            xValue: q,
            yValue: L
          });
        }
        var U = Zb(this.state, this.props.data, this.props.layout, E);
        return U ? N(N({}, _), U) : null;
      }
    }, {
      key: "inRange",
      value: function(b, O) {
        var m = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : 1, x = this.props.layout, _ = b / m, P = O / m;
        if (x === "horizontal" || x === "vertical") {
          var E = this.state.offset, I = _ >= E.left && _ <= E.left + E.width && P >= E.top && P <= E.top + E.height;
          return I ? {
            x: _,
            y: P
          } : null;
        }
        var S = this.state, j = S.angleAxisMap, M = S.radiusAxisMap;
        if (j && M) {
          var R = Wt(j);
          return ig({
            x: _,
            y: P
          }, R);
        }
        return null;
      }
    }, {
      key: "parseEventsOfWrapper",
      value: function() {
        var b = this.props.children, O = this.getTooltipEventType(), m = Qe(b, gt), x = {};
        m && O === "axis" && (m.props.trigger === "click" ? x = {
          onClick: this.handleClick
        } : x = {
          onMouseEnter: this.handleMouseEnter,
          onDoubleClick: this.handleDoubleClick,
          onMouseMove: this.handleMouseMove,
          onMouseLeave: this.handleMouseLeave,
          onTouchMove: this.handleTouchMove,
          onTouchStart: this.handleTouchStart,
          onTouchEnd: this.handleTouchEnd,
          onContextMenu: this.handleContextMenu
        });
        var _ = Gi(this.props, this.handleOuterEvent);
        return N(N({}, _), x);
      }
    }, {
      key: "addListener",
      value: function() {
        _l.on(Sl, this.handleReceiveSyncEvent);
      }
    }, {
      key: "removeListener",
      value: function() {
        _l.removeListener(Sl, this.handleReceiveSyncEvent);
      }
    }, {
      key: "filterFormatItem",
      value: function(b, O, m) {
        for (var x = this.state.formattedGraphicalItems, _ = 0, P = x.length; _ < P; _++) {
          var E = x[_];
          if (E.item === b || E.props.key === b.key || O === Ht(E.item.type) && m === E.childIndex)
            return E;
        }
        return null;
      }
    }, {
      key: "renderClipPath",
      value: function() {
        var b = this.clipPathId, O = this.state.offset, m = O.left, x = O.top, _ = O.height, P = O.width;
        return /* @__PURE__ */ T.createElement("defs", null, /* @__PURE__ */ T.createElement("clipPath", {
          id: b
        }, /* @__PURE__ */ T.createElement("rect", {
          x: m,
          y: x,
          height: _,
          width: P
        })));
      }
    }, {
      key: "getXScales",
      value: function() {
        var b = this.state.xAxisMap;
        return b ? Object.entries(b).reduce(function(O, m) {
          var x = Vb(m, 2), _ = x[0], P = x[1];
          return N(N({}, O), {}, ie({}, _, P.scale));
        }, {}) : null;
      }
    }, {
      key: "getYScales",
      value: function() {
        var b = this.state.yAxisMap;
        return b ? Object.entries(b).reduce(function(O, m) {
          var x = Vb(m, 2), _ = x[0], P = x[1];
          return N(N({}, O), {}, ie({}, _, P.scale));
        }, {}) : null;
      }
    }, {
      key: "getXScaleByAxisId",
      value: function(b) {
        var O;
        return (O = this.state.xAxisMap) === null || O === void 0 || (O = O[b]) === null || O === void 0 ? void 0 : O.scale;
      }
    }, {
      key: "getYScaleByAxisId",
      value: function(b) {
        var O;
        return (O = this.state.yAxisMap) === null || O === void 0 || (O = O[b]) === null || O === void 0 ? void 0 : O.scale;
      }
    }, {
      key: "getItemByXY",
      value: function(b) {
        var O = this.state, m = O.formattedGraphicalItems, x = O.activeItem;
        if (m && m.length)
          for (var _ = 0, P = m.length; _ < P; _++) {
            var E = m[_], I = E.props, S = E.item, j = S.type.defaultProps !== void 0 ? N(N({}, S.type.defaultProps), S.props) : S.props, M = Ht(S.type);
            if (M === "Bar") {
              var R = (I.data || []).find(function(U) {
                return V$(b, U);
              });
              if (R)
                return {
                  graphicalItem: E,
                  payload: R
                };
            } else if (M === "RadialBar") {
              var k = (I.data || []).find(function(U) {
                return ig(b, U);
              });
              if (k)
                return {
                  graphicalItem: E,
                  payload: k
                };
            } else if (po(E, x) || ho(E, x) || ci(E, x)) {
              var q = qR({
                graphicalItem: E,
                activeTooltipItem: x,
                itemData: j.data
              }), L = j.activeIndex === void 0 ? q : j.activeIndex;
              return {
                graphicalItem: N(N({}, E), {}, {
                  childIndex: L
                }),
                payload: ci(E, x) ? j.data[q] : E.props.data[q]
              };
            }
          }
        return null;
      }
    }, {
      key: "render",
      value: function() {
        var b = this;
        if (!mh(this))
          return null;
        var O = this.props, m = O.children, x = O.className, _ = O.width, P = O.height, E = O.style, I = O.compact, S = O.title, j = O.desc, M = Yb(O, cq), R = fe(M, !1);
        if (I)
          return /* @__PURE__ */ T.createElement(jb, {
            state: this.state,
            width: this.props.width,
            height: this.props.height,
            clipPathId: this.clipPathId
          }, /* @__PURE__ */ T.createElement(Ml, Cr({}, R, {
            width: _,
            height: P,
            title: S,
            desc: j
          }), this.renderClipPath(), bh(m, this.renderMap)));
        if (this.props.accessibilityLayer) {
          var k, q;
          R.tabIndex = (k = this.props.tabIndex) !== null && k !== void 0 ? k : 0, R.role = (q = this.props.role) !== null && q !== void 0 ? q : "application", R.onKeyDown = function(U) {
            b.accessibilityManager.keyboardEvent(U);
          }, R.onFocus = function() {
            b.accessibilityManager.focus();
          };
        }
        var L = this.parseEventsOfWrapper();
        return /* @__PURE__ */ T.createElement(jb, {
          state: this.state,
          width: this.props.width,
          height: this.props.height,
          clipPathId: this.clipPathId
        }, /* @__PURE__ */ T.createElement("div", Cr({
          className: pe("recharts-wrapper", x),
          style: N({
            position: "relative",
            cursor: "default",
            width: _,
            height: P
          }, E)
        }, L, {
          ref: function(z) {
            b.container = z;
          }
        }), /* @__PURE__ */ T.createElement(Ml, Cr({}, R, {
          width: _,
          height: P,
          title: S,
          desc: j,
          style: Aq
        }), this.renderClipPath(), bh(m, this.renderMap)), this.renderLegend(), this.renderTooltip()));
      }
    }]);
  }(u0);
  ie(y, "displayName", r), ie(y, "defaultProps", N({
    layout: "horizontal",
    stackOffset: "none",
    barCategoryGap: "10%",
    barGap: 4,
    margin: {
      top: 5,
      right: 5,
      bottom: 5,
      left: 5
    },
    reverseStackOrder: !1,
    syncMethod: "index"
  }, l)), ie(y, "getDerivedStateFromProps", function(h, g) {
    var w = h.dataKey, b = h.data, O = h.children, m = h.width, x = h.height, _ = h.layout, P = h.stackOffset, E = h.margin, I = g.dataStartIndex, S = g.dataEndIndex;
    if (g.updateId === void 0) {
      var j = Jb(h);
      return N(N(N({}, j), {}, {
        updateId: 0
      }, p(N(N({
        props: h
      }, j), {}, {
        updateId: 0
      }), g)), {}, {
        prevDataKey: w,
        prevData: b,
        prevWidth: m,
        prevHeight: x,
        prevLayout: _,
        prevStackOffset: P,
        prevMargin: E,
        prevChildren: O
      });
    }
    if (w !== g.prevDataKey || b !== g.prevData || m !== g.prevWidth || x !== g.prevHeight || _ !== g.prevLayout || P !== g.prevStackOffset || !Mr(E, g.prevMargin)) {
      var M = Jb(h), R = {
        // (chartX, chartY) are (0,0) in default state, but we want to keep the last mouse position to avoid
        // any flickering
        chartX: g.chartX,
        chartY: g.chartY,
        // The tooltip should stay active when it was active in the previous render. If this is not
        // the case, the tooltip disappears and immediately re-appears, causing a flickering effect
        isTooltipActive: g.isTooltipActive
      }, k = N(N({}, Zb(g, b, _)), {}, {
        updateId: g.updateId + 1
      }), q = N(N(N({}, M), R), k);
      return N(N(N({}, q), p(N({
        props: h
      }, q), g)), {}, {
        prevDataKey: w,
        prevData: b,
        prevWidth: m,
        prevHeight: x,
        prevLayout: _,
        prevStackOffset: P,
        prevMargin: E,
        prevChildren: O
      });
    }
    if (!jl(O, g.prevChildren)) {
      var L, U, z, $, D = Qe(O, Kr), B = D && (L = (U = D.props) === null || U === void 0 ? void 0 : U.startIndex) !== null && L !== void 0 ? L : I, G = D && (z = ($ = D.props) === null || $ === void 0 ? void 0 : $.endIndex) !== null && z !== void 0 ? z : S, V = B !== I || G !== S, te = !ce(b), re = te && !V ? g.updateId : g.updateId + 1;
      return N(N({
        updateId: re
      }, p(N(N({
        props: h
      }, g), {}, {
        updateId: re,
        dataStartIndex: B,
        dataEndIndex: G
      }), g)), {}, {
        prevChildren: O,
        dataStartIndex: B,
        dataEndIndex: G
      });
    }
    return null;
  }), ie(y, "renderActiveDot", function(h, g, w) {
    var b;
    return /* @__PURE__ */ xt(h) ? b = /* @__PURE__ */ De(h, g) : ue(h) ? b = h(g) : b = /* @__PURE__ */ T.createElement(Hd, g), /* @__PURE__ */ T.createElement(je, {
      className: "recharts-active-dot",
      key: w
    }, b);
  });
  var v = /* @__PURE__ */ zO(function(g, w) {
    return /* @__PURE__ */ T.createElement(y, Cr({}, g, {
      ref: w
    }));
  });
  return v.displayName = y.displayName, v;
}, Dq = Nq({
  chartName: "LineChart",
  GraphicalChild: tn,
  axisComponents: [{
    axisType: "xAxis",
    AxisComp: xo
  }, {
    axisType: "yAxis",
    AxisComp: wo
  }],
  formatAxisMap: Rk
});
function gO(e, [t, r]) {
  return Math.min(r, Math.max(t, e));
}
function lr(e, t, { checkForDefaultPrevented: r = !0 } = {}) {
  return function(i) {
    if (e == null || e(i), r === !1 || !i.defaultPrevented)
      return t == null ? void 0 : t(i);
  };
}
function e0(e, t) {
  if (typeof e == "function")
    return e(t);
  e != null && (e.current = t);
}
function bO(...e) {
  return (t) => {
    let r = !1;
    const n = e.map((i) => {
      const a = e0(i, t);
      return !r && typeof a == "function" && (r = !0), a;
    });
    if (r)
      return () => {
        for (let i = 0; i < n.length; i++) {
          const a = n[i];
          typeof a == "function" ? a() : e0(e[i], null);
        }
      };
  };
}
function Yt(...e) {
  return Q.useCallback(bO(...e), e);
}
function rp(e, t = []) {
  let r = [];
  function n(a, o) {
    const u = Q.createContext(o), s = r.length;
    r = [...r, o];
    const c = (l) => {
      var g;
      const { scope: d, children: p, ...y } = l, v = ((g = d == null ? void 0 : d[e]) == null ? void 0 : g[s]) || u, h = Q.useMemo(() => y, Object.values(y));
      return /* @__PURE__ */ Y.jsx(v.Provider, { value: h, children: p });
    };
    c.displayName = a + "Provider";
    function f(l, d) {
      var v;
      const p = ((v = d == null ? void 0 : d[e]) == null ? void 0 : v[s]) || u, y = Q.useContext(p);
      if (y) return y;
      if (o !== void 0) return o;
      throw new Error(`\`${l}\` must be used within \`${a}\``);
    }
    return [c, f];
  }
  const i = () => {
    const a = r.map((o) => Q.createContext(o));
    return function(u) {
      const s = (u == null ? void 0 : u[e]) || a;
      return Q.useMemo(
        () => ({ [`__scope${e}`]: { ...u, [e]: s } }),
        [u, s]
      );
    };
  };
  return i.scopeName = e, [n, qq(i, ...t)];
}
function qq(...e) {
  const t = e[0];
  if (e.length === 1) return t;
  const r = () => {
    const n = e.map((i) => ({
      useScope: i(),
      scopeName: i.scopeName
    }));
    return function(a) {
      const o = n.reduce((u, { useScope: s, scopeName: c }) => {
        const l = s(a)[`__scope${c}`];
        return { ...u, ...l };
      }, {});
      return Q.useMemo(() => ({ [`__scope${t.scopeName}`]: o }), [o]);
    };
  };
  return r.scopeName = t.scopeName, r;
}
function xO(e) {
  const t = Q.useRef(e);
  return Q.useEffect(() => {
    t.current = e;
  }), Q.useMemo(() => (...r) => {
    var n;
    return (n = t.current) == null ? void 0 : n.call(t, ...r);
  }, []);
}
function wO({
  prop: e,
  defaultProp: t,
  onChange: r = () => {
  }
}) {
  const [n, i] = Lq({ defaultProp: t, onChange: r }), a = e !== void 0, o = a ? e : n, u = xO(r), s = Q.useCallback(
    (c) => {
      if (a) {
        const l = typeof c == "function" ? c(e) : c;
        l !== e && u(l);
      } else
        i(c);
    },
    [a, e, i, u]
  );
  return [o, s];
}
function Lq({
  defaultProp: e,
  onChange: t
}) {
  const r = Q.useState(e), [n] = r, i = Q.useRef(n), a = xO(t);
  return Q.useEffect(() => {
    i.current !== n && (a(n), i.current = n);
  }, [n, i, a]), r;
}
var Bq = Q.createContext(void 0);
function Fq(e) {
  const t = Q.useContext(Bq);
  return e || t || "ltr";
}
function OO(e) {
  const t = Q.useRef({ value: e, previous: e });
  return Q.useMemo(() => (t.current.value !== e && (t.current.previous = t.current.value, t.current.value = e), t.current.previous), [e]);
}
var zq = globalThis != null && globalThis.document ? Q.useLayoutEffect : () => {
};
function _O(e) {
  const [t, r] = Q.useState(void 0);
  return zq(() => {
    if (e) {
      r({ width: e.offsetWidth, height: e.offsetHeight });
      const n = new ResizeObserver((i) => {
        if (!Array.isArray(i) || !i.length)
          return;
        const a = i[0];
        let o, u;
        if ("borderBoxSize" in a) {
          const s = a.borderBoxSize, c = Array.isArray(s) ? s[0] : s;
          o = c.inlineSize, u = c.blockSize;
        } else
          o = e.offsetWidth, u = e.offsetHeight;
        r({ width: o, height: u });
      });
      return n.observe(e, { box: "border-box" }), () => n.unobserve(e);
    } else
      r(void 0);
  }, [e]), t;
}
var za = Q.forwardRef((e, t) => {
  const { children: r, ...n } = e, i = Q.Children.toArray(r), a = i.find(Wq);
  if (a) {
    const o = a.props.children, u = i.map((s) => s === a ? Q.Children.count(o) > 1 ? Q.Children.only(null) : Q.isValidElement(o) ? o.props.children : null : s);
    return /* @__PURE__ */ Y.jsx(Hf, { ...n, ref: t, children: Q.isValidElement(o) ? Q.cloneElement(o, void 0, u) : null });
  }
  return /* @__PURE__ */ Y.jsx(Hf, { ...n, ref: t, children: r });
});
za.displayName = "Slot";
var Hf = Q.forwardRef((e, t) => {
  const { children: r, ...n } = e;
  if (Q.isValidElement(r)) {
    const i = Hq(r);
    return Q.cloneElement(r, {
      ...Gq(n, r.props),
      // @ts-ignore
      ref: t ? bO(t, i) : i
    });
  }
  return Q.Children.count(r) > 1 ? Q.Children.only(null) : null;
});
Hf.displayName = "SlotClone";
var Uq = ({ children: e }) => /* @__PURE__ */ Y.jsx(Y.Fragment, { children: e });
function Wq(e) {
  return Q.isValidElement(e) && e.type === Uq;
}
function Gq(e, t) {
  const r = { ...t };
  for (const n in t) {
    const i = e[n], a = t[n];
    /^on[A-Z]/.test(n) ? i && a ? r[n] = (...u) => {
      a(...u), i(...u);
    } : i && (r[n] = i) : n === "style" ? r[n] = { ...i, ...a } : n === "className" && (r[n] = [i, a].filter(Boolean).join(" "));
  }
  return { ...e, ...r };
}
function Hq(e) {
  var n, i;
  let t = (n = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : n.get, r = t && "isReactWarning" in t && t.isReactWarning;
  return r ? e.ref : (t = (i = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : i.get, r = t && "isReactWarning" in t && t.isReactWarning, r ? e.props.ref : e.props.ref || e.ref);
}
var Kq = [
  "a",
  "button",
  "div",
  "form",
  "h2",
  "h3",
  "img",
  "input",
  "label",
  "li",
  "nav",
  "ol",
  "p",
  "span",
  "svg",
  "ul"
], hn = Kq.reduce((e, t) => {
  const r = Q.forwardRef((n, i) => {
    const { asChild: a, ...o } = n, u = a ? za : t;
    return typeof window < "u" && (window[Symbol.for("radix-ui")] = !0), /* @__PURE__ */ Y.jsx(u, { ...o, ref: i });
  });
  return r.displayName = `Primitive.${t}`, { ...e, [t]: r };
}, {});
function Vq(e) {
  const t = e + "CollectionProvider", [r, n] = rp(t), [i, a] = r(
    t,
    { collectionRef: { current: null }, itemMap: /* @__PURE__ */ new Map() }
  ), o = (p) => {
    const { scope: y, children: v } = p, h = T.useRef(null), g = T.useRef(/* @__PURE__ */ new Map()).current;
    return /* @__PURE__ */ Y.jsx(i, { scope: y, itemMap: g, collectionRef: h, children: v });
  };
  o.displayName = t;
  const u = e + "CollectionSlot", s = T.forwardRef(
    (p, y) => {
      const { scope: v, children: h } = p, g = a(u, v), w = Yt(y, g.collectionRef);
      return /* @__PURE__ */ Y.jsx(za, { ref: w, children: h });
    }
  );
  s.displayName = u;
  const c = e + "CollectionItemSlot", f = "data-radix-collection-item", l = T.forwardRef(
    (p, y) => {
      const { scope: v, children: h, ...g } = p, w = T.useRef(null), b = Yt(y, w), O = a(c, v);
      return T.useEffect(() => (O.itemMap.set(w, { ref: w, ...g }), () => void O.itemMap.delete(w))), /* @__PURE__ */ Y.jsx(za, { [f]: "", ref: b, children: h });
    }
  );
  l.displayName = c;
  function d(p) {
    const y = a(e + "CollectionConsumer", p);
    return T.useCallback(() => {
      const h = y.collectionRef.current;
      if (!h) return [];
      const g = Array.from(h.querySelectorAll(`[${f}]`));
      return Array.from(y.itemMap.values()).sort(
        (O, m) => g.indexOf(O.ref.current) - g.indexOf(m.ref.current)
      );
    }, [y.collectionRef, y.itemMap]);
  }
  return [
    { Provider: o, Slot: s, ItemSlot: l },
    d,
    n
  ];
}
var SO = ["PageUp", "PageDown"], PO = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"], AO = {
  "from-left": ["Home", "PageDown", "ArrowDown", "ArrowLeft"],
  "from-right": ["Home", "PageDown", "ArrowDown", "ArrowRight"],
  "from-bottom": ["Home", "PageDown", "ArrowDown", "ArrowLeft"],
  "from-top": ["Home", "PageDown", "ArrowUp", "ArrowLeft"]
}, vn = "Slider", [Kf, Yq, Xq] = Vq(vn), [EO, g2] = rp(vn, [
  Xq
]), [Zq, _o] = EO(vn), TO = Q.forwardRef(
  (e, t) => {
    const {
      name: r,
      min: n = 0,
      max: i = 100,
      step: a = 1,
      orientation: o = "horizontal",
      disabled: u = !1,
      minStepsBetweenThumbs: s = 0,
      defaultValue: c = [n],
      value: f,
      onValueChange: l = () => {
      },
      onValueCommit: d = () => {
      },
      inverted: p = !1,
      form: y,
      ...v
    } = e, h = Q.useRef(/* @__PURE__ */ new Set()), g = Q.useRef(0), b = o === "horizontal" ? Jq : Qq, [O = [], m] = wO({
      prop: f,
      defaultProp: c,
      onChange: (S) => {
        var M;
        (M = [...h.current][g.current]) == null || M.focus(), l(S);
      }
    }), x = Q.useRef(O);
    function _(S) {
      const j = i2(O, S);
      I(S, j);
    }
    function P(S) {
      I(S, g.current);
    }
    function E() {
      const S = x.current[g.current];
      O[g.current] !== S && d(O);
    }
    function I(S, j, { commit: M } = { commit: !1 }) {
      const R = s2(a), k = c2(Math.round((S - n) / a) * a + n, R), q = gO(k, [n, i]);
      m((L = []) => {
        const U = r2(L, q, j);
        if (u2(U, s * a)) {
          g.current = U.indexOf(q);
          const z = String(U) !== String(L);
          return z && M && d(U), z ? U : L;
        } else
          return L;
      });
    }
    return /* @__PURE__ */ Y.jsx(
      Zq,
      {
        scope: e.__scopeSlider,
        name: r,
        disabled: u,
        min: n,
        max: i,
        valueIndexToChangeRef: g,
        thumbs: h.current,
        values: O,
        orientation: o,
        form: y,
        children: /* @__PURE__ */ Y.jsx(Kf.Provider, { scope: e.__scopeSlider, children: /* @__PURE__ */ Y.jsx(Kf.Slot, { scope: e.__scopeSlider, children: /* @__PURE__ */ Y.jsx(
          b,
          {
            "aria-disabled": u,
            "data-disabled": u ? "" : void 0,
            ...v,
            ref: t,
            onPointerDown: lr(v.onPointerDown, () => {
              u || (x.current = O);
            }),
            min: n,
            max: i,
            inverted: p,
            onSlideStart: u ? void 0 : _,
            onSlideMove: u ? void 0 : P,
            onSlideEnd: u ? void 0 : E,
            onHomeKeyDown: () => !u && I(n, 0, { commit: !0 }),
            onEndKeyDown: () => !u && I(i, O.length - 1, { commit: !0 }),
            onStepKeyDown: ({ event: S, direction: j }) => {
              if (!u) {
                const k = SO.includes(S.key) || S.shiftKey && PO.includes(S.key) ? 10 : 1, q = g.current, L = O[q], U = a * k * j;
                I(L + U, q, { commit: !0 });
              }
            }
          }
        ) }) })
      }
    );
  }
);
TO.displayName = vn;
var [jO, CO] = EO(vn, {
  startEdge: "left",
  endEdge: "right",
  size: "width",
  direction: 1
}), Jq = Q.forwardRef(
  (e, t) => {
    const {
      min: r,
      max: n,
      dir: i,
      inverted: a,
      onSlideStart: o,
      onSlideMove: u,
      onSlideEnd: s,
      onStepKeyDown: c,
      ...f
    } = e, [l, d] = Q.useState(null), p = Yt(t, (b) => d(b)), y = Q.useRef(void 0), v = Fq(i), h = v === "ltr", g = h && !a || !h && a;
    function w(b) {
      const O = y.current || l.getBoundingClientRect(), m = [0, O.width], _ = np(m, g ? [r, n] : [n, r]);
      return y.current = O, _(b - O.left);
    }
    return /* @__PURE__ */ Y.jsx(
      jO,
      {
        scope: e.__scopeSlider,
        startEdge: g ? "left" : "right",
        endEdge: g ? "right" : "left",
        direction: g ? 1 : -1,
        size: "width",
        children: /* @__PURE__ */ Y.jsx(
          MO,
          {
            dir: v,
            "data-orientation": "horizontal",
            ...f,
            ref: p,
            style: {
              ...f.style,
              "--radix-slider-thumb-transform": "translateX(-50%)"
            },
            onSlideStart: (b) => {
              const O = w(b.clientX);
              o == null || o(O);
            },
            onSlideMove: (b) => {
              const O = w(b.clientX);
              u == null || u(O);
            },
            onSlideEnd: () => {
              y.current = void 0, s == null || s();
            },
            onStepKeyDown: (b) => {
              const m = AO[g ? "from-left" : "from-right"].includes(b.key);
              c == null || c({ event: b, direction: m ? -1 : 1 });
            }
          }
        )
      }
    );
  }
), Qq = Q.forwardRef(
  (e, t) => {
    const {
      min: r,
      max: n,
      inverted: i,
      onSlideStart: a,
      onSlideMove: o,
      onSlideEnd: u,
      onStepKeyDown: s,
      ...c
    } = e, f = Q.useRef(null), l = Yt(t, f), d = Q.useRef(void 0), p = !i;
    function y(v) {
      const h = d.current || f.current.getBoundingClientRect(), g = [0, h.height], b = np(g, p ? [n, r] : [r, n]);
      return d.current = h, b(v - h.top);
    }
    return /* @__PURE__ */ Y.jsx(
      jO,
      {
        scope: e.__scopeSlider,
        startEdge: p ? "bottom" : "top",
        endEdge: p ? "top" : "bottom",
        size: "height",
        direction: p ? 1 : -1,
        children: /* @__PURE__ */ Y.jsx(
          MO,
          {
            "data-orientation": "vertical",
            ...c,
            ref: l,
            style: {
              ...c.style,
              "--radix-slider-thumb-transform": "translateY(50%)"
            },
            onSlideStart: (v) => {
              const h = y(v.clientY);
              a == null || a(h);
            },
            onSlideMove: (v) => {
              const h = y(v.clientY);
              o == null || o(h);
            },
            onSlideEnd: () => {
              d.current = void 0, u == null || u();
            },
            onStepKeyDown: (v) => {
              const g = AO[p ? "from-bottom" : "from-top"].includes(v.key);
              s == null || s({ event: v, direction: g ? -1 : 1 });
            }
          }
        )
      }
    );
  }
), MO = Q.forwardRef(
  (e, t) => {
    const {
      __scopeSlider: r,
      onSlideStart: n,
      onSlideMove: i,
      onSlideEnd: a,
      onHomeKeyDown: o,
      onEndKeyDown: u,
      onStepKeyDown: s,
      ...c
    } = e, f = _o(vn, r);
    return /* @__PURE__ */ Y.jsx(
      hn.span,
      {
        ...c,
        ref: t,
        onKeyDown: lr(e.onKeyDown, (l) => {
          l.key === "Home" ? (o(l), l.preventDefault()) : l.key === "End" ? (u(l), l.preventDefault()) : SO.concat(PO).includes(l.key) && (s(l), l.preventDefault());
        }),
        onPointerDown: lr(e.onPointerDown, (l) => {
          const d = l.target;
          d.setPointerCapture(l.pointerId), l.preventDefault(), f.thumbs.has(d) ? d.focus() : n(l);
        }),
        onPointerMove: lr(e.onPointerMove, (l) => {
          l.target.hasPointerCapture(l.pointerId) && i(l);
        }),
        onPointerUp: lr(e.onPointerUp, (l) => {
          const d = l.target;
          d.hasPointerCapture(l.pointerId) && (d.releasePointerCapture(l.pointerId), a(l));
        })
      }
    );
  }
), IO = "SliderTrack", $O = Q.forwardRef(
  (e, t) => {
    const { __scopeSlider: r, ...n } = e, i = _o(IO, r);
    return /* @__PURE__ */ Y.jsx(
      hn.span,
      {
        "data-disabled": i.disabled ? "" : void 0,
        "data-orientation": i.orientation,
        ...n,
        ref: t
      }
    );
  }
);
$O.displayName = IO;
var Vf = "SliderRange", RO = Q.forwardRef(
  (e, t) => {
    const { __scopeSlider: r, ...n } = e, i = _o(Vf, r), a = CO(Vf, r), o = Q.useRef(null), u = Yt(t, o), s = i.values.length, c = i.values.map(
      (d) => NO(d, i.min, i.max)
    ), f = s > 1 ? Math.min(...c) : 0, l = 100 - Math.max(...c);
    return /* @__PURE__ */ Y.jsx(
      hn.span,
      {
        "data-orientation": i.orientation,
        "data-disabled": i.disabled ? "" : void 0,
        ...n,
        ref: u,
        style: {
          ...e.style,
          [a.startEdge]: f + "%",
          [a.endEdge]: l + "%"
        }
      }
    );
  }
);
RO.displayName = Vf;
var Yf = "SliderThumb", kO = Q.forwardRef(
  (e, t) => {
    const r = Yq(e.__scopeSlider), [n, i] = Q.useState(null), a = Yt(t, (u) => i(u)), o = Q.useMemo(
      () => n ? r().findIndex((u) => u.ref.current === n) : -1,
      [r, n]
    );
    return /* @__PURE__ */ Y.jsx(e2, { ...e, ref: a, index: o });
  }
), e2 = Q.forwardRef(
  (e, t) => {
    const { __scopeSlider: r, index: n, name: i, ...a } = e, o = _o(Yf, r), u = CO(Yf, r), [s, c] = Q.useState(null), f = Yt(t, (w) => c(w)), l = s ? o.form || !!s.closest("form") : !0, d = _O(s), p = o.values[n], y = p === void 0 ? 0 : NO(p, o.min, o.max), v = n2(n, o.values.length), h = d == null ? void 0 : d[u.size], g = h ? a2(h, y, u.direction) : 0;
    return Q.useEffect(() => {
      if (s)
        return o.thumbs.add(s), () => {
          o.thumbs.delete(s);
        };
    }, [s, o.thumbs]), /* @__PURE__ */ Y.jsxs(
      "span",
      {
        style: {
          transform: "var(--radix-slider-thumb-transform)",
          position: "absolute",
          [u.startEdge]: `calc(${y}% + ${g}px)`
        },
        children: [
          /* @__PURE__ */ Y.jsx(Kf.ItemSlot, { scope: e.__scopeSlider, children: /* @__PURE__ */ Y.jsx(
            hn.span,
            {
              role: "slider",
              "aria-label": e["aria-label"] || v,
              "aria-valuemin": o.min,
              "aria-valuenow": p,
              "aria-valuemax": o.max,
              "aria-orientation": o.orientation,
              "data-orientation": o.orientation,
              "data-disabled": o.disabled ? "" : void 0,
              tabIndex: o.disabled ? void 0 : 0,
              ...a,
              ref: f,
              style: p === void 0 ? { display: "none" } : e.style,
              onFocus: lr(e.onFocus, () => {
                o.valueIndexToChangeRef.current = n;
              })
            }
          ) }),
          l && /* @__PURE__ */ Y.jsx(
            t2,
            {
              name: i ?? (o.name ? o.name + (o.values.length > 1 ? "[]" : "") : void 0),
              form: o.form,
              value: p
            },
            n
          )
        ]
      }
    );
  }
);
kO.displayName = Yf;
var t2 = (e) => {
  const { value: t, ...r } = e, n = Q.useRef(null), i = OO(t);
  return Q.useEffect(() => {
    const a = n.current, o = window.HTMLInputElement.prototype, s = Object.getOwnPropertyDescriptor(o, "value").set;
    if (i !== t && s) {
      const c = new Event("input", { bubbles: !0 });
      s.call(a, t), a.dispatchEvent(c);
    }
  }, [i, t]), /* @__PURE__ */ Y.jsx("input", { style: { display: "none" }, ...r, ref: n, defaultValue: t });
};
function r2(e = [], t, r) {
  const n = [...e];
  return n[r] = t, n.sort((i, a) => i - a);
}
function NO(e, t, r) {
  const a = 100 / (r - t) * (e - t);
  return gO(a, [0, 100]);
}
function n2(e, t) {
  return t > 2 ? `Value ${e + 1} of ${t}` : t === 2 ? ["Minimum", "Maximum"][e] : void 0;
}
function i2(e, t) {
  if (e.length === 1) return 0;
  const r = e.map((i) => Math.abs(i - t)), n = Math.min(...r);
  return r.indexOf(n);
}
function a2(e, t, r) {
  const n = e / 2, a = np([0, 50], [0, n]);
  return (n - a(t) * r) * r;
}
function o2(e) {
  return e.slice(0, -1).map((t, r) => e[r + 1] - t);
}
function u2(e, t) {
  if (t > 0) {
    const r = o2(e);
    return Math.min(...r) >= t;
  }
  return !0;
}
function np(e, t) {
  return (r) => {
    if (e[0] === e[1] || t[0] === t[1]) return t[0];
    const n = (t[1] - t[0]) / (e[1] - e[0]);
    return t[0] + n * (r - e[0]);
  };
}
function s2(e) {
  return (String(e).split(".")[1] || "").length;
}
function c2(e, t) {
  const r = Math.pow(10, t);
  return Math.round(e * r) / r;
}
var t0 = TO, r0 = $O, n0 = RO, i0 = kO, ip = "Switch", [l2, b2] = rp(ip), [f2, d2] = l2(ip), DO = Q.forwardRef(
  (e, t) => {
    const {
      __scopeSwitch: r,
      name: n,
      checked: i,
      defaultChecked: a,
      required: o,
      disabled: u,
      value: s = "on",
      onCheckedChange: c,
      form: f,
      ...l
    } = e, [d, p] = Q.useState(null), y = Yt(t, (b) => p(b)), v = Q.useRef(!1), h = d ? f || !!d.closest("form") : !0, [g = !1, w] = wO({
      prop: i,
      defaultProp: a,
      onChange: c
    });
    return /* @__PURE__ */ Y.jsxs(f2, { scope: r, checked: g, disabled: u, children: [
      /* @__PURE__ */ Y.jsx(
        hn.button,
        {
          type: "button",
          role: "switch",
          "aria-checked": g,
          "aria-required": o,
          "data-state": BO(g),
          "data-disabled": u ? "" : void 0,
          disabled: u,
          value: s,
          ...l,
          ref: y,
          onClick: lr(e.onClick, (b) => {
            w((O) => !O), h && (v.current = b.isPropagationStopped(), v.current || b.stopPropagation());
          })
        }
      ),
      h && /* @__PURE__ */ Y.jsx(
        p2,
        {
          control: d,
          bubbles: !v.current,
          name: n,
          value: s,
          checked: g,
          required: o,
          disabled: u,
          form: f,
          style: { transform: "translateX(-100%)" }
        }
      )
    ] });
  }
);
DO.displayName = ip;
var qO = "SwitchThumb", LO = Q.forwardRef(
  (e, t) => {
    const { __scopeSwitch: r, ...n } = e, i = d2(qO, r);
    return /* @__PURE__ */ Y.jsx(
      hn.span,
      {
        "data-state": BO(i.checked),
        "data-disabled": i.disabled ? "" : void 0,
        ...n,
        ref: t
      }
    );
  }
);
LO.displayName = qO;
var p2 = (e) => {
  const { control: t, checked: r, bubbles: n = !0, ...i } = e, a = Q.useRef(null), o = OO(r), u = _O(t);
  return Q.useEffect(() => {
    const s = a.current, c = window.HTMLInputElement.prototype, l = Object.getOwnPropertyDescriptor(c, "checked").set;
    if (o !== r && l) {
      const d = new Event("click", { bubbles: n });
      l.call(s, r), s.dispatchEvent(d);
    }
  }, [o, r, n]), /* @__PURE__ */ Y.jsx(
    "input",
    {
      type: "checkbox",
      "aria-hidden": !0,
      defaultChecked: r,
      ...i,
      tabIndex: -1,
      ref: a,
      style: {
        ...e.style,
        ...u,
        position: "absolute",
        pointerEvents: "none",
        opacity: 0,
        margin: 0
      }
    }
  );
};
function BO(e) {
  return e ? "checked" : "unchecked";
}
var h2 = DO, v2 = LO;
const x2 = () => {
  const [e, t] = Er(0.5), [r, n] = Er(1), [i, a] = Er(!1), [o, u] = Er([]), s = "#E6425E", c = "#1D84B5";
  Xf(() => {
    let l = [], d = 1;
    for (let p = 0; p < 50; p++)
      l.push({
        time: p,
        value: d,
        input: r
      }), d = i ? Math.tanh(e * d + r) : e * d + r;
    u(l);
  }, [e, r, i]);
  const f = () => /* @__PURE__ */ Y.jsxs("svg", { viewBox: "0 0 200 120", className: "w-full h-32", children: [
    /* @__PURE__ */ Y.jsx("rect", { x: "70", y: "30", width: "60", height: "40", fill: "white", stroke: s, strokeWidth: "2" }),
    /* @__PURE__ */ Y.jsx("text", { x: "100", y: "50", textAnchor: "middle", dominantBaseline: "middle", className: "text-sm", children: i ? "tanh(h)" : "h" }),
    /* @__PURE__ */ Y.jsx(
      "path",
      {
        d: "M100 30 C100 10, 100 10, 100 10 L100 10 C100 10, 100 30, 100 30",
        fill: "none",
        stroke: s,
        strokeWidth: "2"
      }
    ),
    /* @__PURE__ */ Y.jsx("line", { x1: "40", y1: "50", x2: "70", y2: "50", stroke: c, strokeWidth: "2", markerEnd: "url(#arrowhead)" }),
    /* @__PURE__ */ Y.jsx("text", { x: "30", y: "50", textAnchor: "middle", className: "text-xs", children: "u" }),
    /* @__PURE__ */ Y.jsx("defs", { children: /* @__PURE__ */ Y.jsx(
      "marker",
      {
        id: "arrowhead",
        markerWidth: "10",
        markerHeight: "7",
        refX: "9",
        refY: "3.5",
        orient: "auto",
        children: /* @__PURE__ */ Y.jsx("polygon", { points: "0 0, 10 3.5, 0 7", fill: c })
      }
    ) }),
    /* @__PURE__ */ Y.jsxs("text", { x: "100", y: "15", textAnchor: "middle", className: "text-xs", children: [
      "w = ",
      e.toFixed(2)
    ] })
  ] });
  return /* @__PURE__ */ Y.jsxs("div", { className: "p-6 space-y-6 bg-white rounded-lg shadow-md", children: [
    /* @__PURE__ */ Y.jsxs("div", { className: "prose max-w-none mb-4", children: [
      /* @__PURE__ */ Y.jsx("h2", { className: "text-xl font-bold mb-2", children: "Simple RNN" }),
      /* @__PURE__ */ Y.jsxs("p", { children: [
        "Equation: ",
        i ? "h(t) = tanh(w * h(t-1) + u)" : "h(t) = w * h(t-1) + u"
      ] }),
      /* @__PURE__ */ Y.jsxs("p", { children: [
        "System behavior: |w| < 1 stable, |w| > 1 unstable ",
        i && ", tanh bounds output to [-1,1]"
      ] })
    ] }),
    /* @__PURE__ */ Y.jsxs("div", { className: "space-y-4", children: [
      /* @__PURE__ */ Y.jsxs("div", { children: [
        /* @__PURE__ */ Y.jsxs("h3", { className: "text-lg font-medium mb-2", children: [
          "Recurrent Weight: ",
          e.toFixed(2)
        ] }),
        /* @__PURE__ */ Y.jsxs(
          t0,
          {
            value: [e],
            onValueChange: ([l]) => t(l),
            min: -1.5,
            max: 1.5,
            step: 0.01,
            className: "slider-root",
            children: [
              /* @__PURE__ */ Y.jsx(r0, { className: "slider-track", children: /* @__PURE__ */ Y.jsx(n0, { className: "slider-range" }) }),
              /* @__PURE__ */ Y.jsx(i0, { className: "slider-thumb" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ Y.jsxs("div", { children: [
        /* @__PURE__ */ Y.jsxs("h3", { className: "text-lg font-medium mb-2", children: [
          "Input Signal: ",
          r.toFixed(2)
        ] }),
        /* @__PURE__ */ Y.jsxs(
          t0,
          {
            value: [r],
            onValueChange: ([l]) => n(l),
            min: -2,
            max: 2,
            step: 0.1,
            className: "slider-root",
            children: [
              /* @__PURE__ */ Y.jsx(r0, { className: "slider-track", children: /* @__PURE__ */ Y.jsx(n0, { className: "slider-range" }) }),
              /* @__PURE__ */ Y.jsx(i0, { className: "slider-thumb" })
            ]
          }
        )
      ] }),
      /* @__PURE__ */ Y.jsxs("div", { className: "flex items-center space-x-2", children: [
        /* @__PURE__ */ Y.jsx(
          h2,
          {
            checked: i,
            onCheckedChange: a,
            className: "switch-root",
            children: /* @__PURE__ */ Y.jsx(v2, { className: "switch-thumb" })
          }
        ),
        /* @__PURE__ */ Y.jsx("label", { className: "text-lg font-medium", children: "Use tanh non-linearity" })
      ] })
    ] }),
    /* @__PURE__ */ Y.jsx(f, {}),
    /* @__PURE__ */ Y.jsx("div", { className: "mt-6 h-64", children: /* @__PURE__ */ Y.jsxs(
      Dq,
      {
        width: 600,
        height: 200,
        data: o,
        margin: { top: 5, right: 20, bottom: 5, left: 0 },
        children: [
          /* @__PURE__ */ Y.jsx(rO, { strokeDasharray: "3 3" }),
          /* @__PURE__ */ Y.jsx(xo, { dataKey: "time" }),
          /* @__PURE__ */ Y.jsx(wo, {}),
          /* @__PURE__ */ Y.jsx(gt, {}),
          /* @__PURE__ */ Y.jsx(tn, { type: "monotone", dataKey: "value", stroke: s, dot: !1, name: "Hidden State", strokeWidth: 2 }),
          /* @__PURE__ */ Y.jsx(tn, { type: "monotone", dataKey: "input", stroke: c, dot: !1, name: "Input", strokeWidth: 2 })
        ]
      }
    ) })
  ] });
};
export {
  x2 as RNNVisualization
};
