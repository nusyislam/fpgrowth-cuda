-- crossref.lua - numbered Word captions and cross-references for report.md.
--
--   ![caption](file.png){#fig:name}     figure, captioned "Figure N: caption"
--   ::: {#tbl:name} <table> :::         table,  captioned "Table N: caption"
--   # Heading {#sec:name}               section (numbered by --number-sections)
--   [@fig:name] [@tbl:name] [@sec:name] -> "Figure N", "Table N", "Section N.M"
--
-- Caption numbers are Word SEQ fields (as Insert Caption makes them), so Word
-- can renumber them and build a table of figures.  The field results are
-- filled in here, so the document is correct without updating fields.
-- Unknown labels stop the build.

local fig, tbl, sec = {}, {}, {}
local nfig, ntbl = 0, 0

local function seq(kind, n)
  return pandoc.RawInline("openxml",
    '<w:r><w:t xml:space="preserve">' .. kind .. ' </w:t></w:r>' ..
    '<w:fldSimple w:instr=" SEQ ' .. kind .. ' \\* ARABIC "><w:r><w:t>' .. n ..
    '</w:t></w:r></w:fldSimple><w:r><w:t xml:space="preserve">: </w:t></w:r>')
end

local function prefix(caption, kind, n)
  local first = caption.long[1]
  if first and (first.t == "Plain" or first.t == "Para") then
    first.content:insert(1, seq(kind, n))
  else
    caption.long:insert(1, pandoc.Plain({ seq(kind, n) }))
  end
end

-- Pass 1: number everything in document order.
local function number(doc)
  local h = { 0, 0 }
  doc:walk({
    Header = function(el)
      if el.classes:includes("unnumbered") or el.level > 2 then return end
      h[el.level] = h[el.level] + 1
      if el.level == 1 then h[2] = 0 end
      if el.identifier:match("^sec:") then
        sec[el.identifier] = el.level == 1 and tostring(h[1]) or (h[1] .. "." .. h[2])
      end
    end,
    Figure = function(el)
      nfig = nfig + 1
      if el.identifier ~= "" then fig[el.identifier] = nfig end
    end,
    Div = function(el)
      if el.identifier:match("^tbl:") then
        ntbl = ntbl + 1
        tbl[el.identifier] = ntbl
      end
    end,
  })
end

local function label(id)
  if fig[id] then return "Figure " .. fig[id] end
  if tbl[id] then return "Table " .. tbl[id] end
  if sec[id] then return "Section " .. sec[id] end
  error("crossref.lua: unknown label @" .. id)
end

function Pandoc(doc)
  number(doc)
  local n = 0
  doc = doc:walk({
    Figure = function(el)
      n = n + 1
      prefix(el.caption, "Figure", n)
      return el
    end,
    Div = function(el)
      if not el.identifier:match("^tbl:") then return end
      local k = tbl[el.identifier]
      return el:walk({ Table = function(t) prefix(t.caption, "Table", k); return t end })
    end,
    Cite = function(el)
      local out = pandoc.List()
      for i, c in ipairs(el.citations) do
        if i > 1 then out:insert(pandoc.Str(", ")) end
        out:insert(pandoc.Str(label(c.id)))
      end
      return out
    end,
  })
  return doc
end
