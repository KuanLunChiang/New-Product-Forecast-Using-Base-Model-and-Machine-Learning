

select rp.RecipeId ,sum(mi.Quantity) as total_sales, ps.date
from dbo.pos_ordersale as ps
inner join dbo.menuitem as mi on ps.MD5KEY_ORDERSALE = mi.MD5KEY_ORDERSALE
inner join dbo.menu_items as mis on mi.Id = mis.MenuItemID
inner join dbo.recipes as rp on rp.RecipeId = mis.RecipeID
group by ps.date,rp.RecipeId;


with cte as(
	select ria.RecipeId, ig.IngredientName, sum(ria.Quantity) as Quantity
	from dbo.recipes_ingredient_assignments as ria
	inner join dbo.ingredients as ig on ria.IngredientId = ig.IngredientId
	group by ig.IngredientName, ria.RecipeId
	union
	select rsa.RecipeId, ig.IngredientName, sum(sia.Quantity) as Quantity
	from dbo.sub_recipe_ingr_assignments as sia
	inner join dbo.ingredients as ig on sia.IngredientId = ig.IngredientId
	inner join dbo.recipe_sub_recipe_assignments as rsa on rsa.SubRecipeID = sia.SubRecipeId
	where sia.Quantity >0
	group by ig.IngredientName, rsa.RecipeId)
select cte.RecipeId, cte.IngredientName, sum(cte.Quantity) as Quantity
from cte
group by cte.RecipeId, cte.IngredientName;


select * from dbo.vw_recipe_ingredient_qty order by recipeID;

select * from dbo.vw_salesNum;