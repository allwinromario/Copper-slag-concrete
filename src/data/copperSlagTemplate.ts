/**
 * Starter template for a copper-slag concrete dataset.
 *
 * The header matches the schema this app expects. The example rows are
 * placeholders in plausible ranges drawn from typical copper-slag concrete
 * studies (Mithun & Narasimhan 2016, Ambily et al. 2015, Khanzadi & Behnood
 * 2009). They are NOT extracted from those papers verbatim — replace them
 * with your own laboratory measurements before training a real model.
 *
 * Required columns: Cement, Water, Fine aggregate, Coarse aggregate,
 * Curing days, Compressive strength.
 *
 * Optional columns (omit or leave blank if you don't have them; the app
 * defaults the inputs and skips the outputs):
 *   - Copper slag %, SG cement / fine / coarse, Fineness modulus
 *   - Split tensile, Flexural strength, Modulus of elasticity, Density
 */
export const COPPER_SLAG_TEMPLATE_CSV = `Cement,Water,Fine aggregate,Coarse aggregate,Copper slag %,Curing days,SG cement,SG fine,SG coarse,Fineness modulus,Compressive strength,Split tensile,Flexural strength,Modulus of elasticity,Density
360,180,720,1180,0,28,3.15,2.65,2.72,2.70,38.0,3.8,4.4,29.5,2380
360,180,684,1180,5,28,3.15,2.78,2.72,2.72,40.5,4.0,4.6,30.4,2395
360,180,648,1180,10,28,3.15,2.92,2.72,2.74,42.8,4.2,4.8,31.2,2410
360,180,612,1180,15,28,3.15,3.07,2.72,2.76,44.2,4.4,5.0,31.8,2425
360,180,576,1180,20,28,3.15,3.21,2.72,2.78,45.0,4.5,5.1,32.1,2440
360,180,540,1180,25,28,3.15,3.36,2.72,2.80,44.6,4.4,5.0,31.9,2455
360,180,504,1180,30,28,3.15,3.50,2.72,2.82,42.0,4.2,4.8,31.0,2470
360,180,468,1180,35,28,3.15,3.65,2.72,2.84,38.5,3.9,4.4,29.7,2485
360,180,432,1180,40,28,3.15,3.79,2.72,2.86,34.0,3.6,4.0,28.0,2500
360,180,396,1180,50,28,3.15,4.08,2.72,2.90,28.5,3.2,3.5,25.5,2530
`;
